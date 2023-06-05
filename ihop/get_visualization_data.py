import gensim.models as gm
import joblib
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import json
import math
import argparse
import os

NSFW_ID = 138
ROOT_ID = 198
BASE_MONTH = "2021-04"
NEW_CLUSTER_ID = -1
NEW_CLUSTER_LABEL = "New Cluster"
BASE_COLOR = "#808080"
SIMILARITY_THRESHOLD = 0.1
SIMILARITY_THRESHOLD_REDUCE = 0.025

class Tree:
    def __init__(self, clusters_df, tsne_df, subreddit_comment_counts_df, color_df, cluster_centers, subreddit_centers, month):
        self.level_to_clusters = {}
        self.subreddits = self.__get_subreddits(clusters_df, tsne_df, subreddit_comment_counts_df, subreddit_centers, month)
        self.kmeans_cluster_node_id_to_cluster = self.__get_kmeans_cluster_node_id_to_subreddits(clusters_df[month + "_kmeans_clusters"].to_numpy(), self.subreddits, cluster_centers)
        if month == BASE_MONTH:
            self.kmeans_cluster_node_id_to_cluster[NEW_CLUSTER_ID] = Cluster(NEW_CLUSTER_ID, None)
            self.kmeans_cluster_node_id_to_cluster[NEW_CLUSTER_ID].taxonomy_label = NEW_CLUSTER_LABEL
            self.kmeans_cluster_node_id_to_cluster[NEW_CLUSTER_ID].level = 1
            self.__num_of_kmeans_clusters = len(self.kmeans_cluster_node_id_to_cluster) - 1
            self.__max_hierarchical_cluster_node_id = (self.__num_of_kmeans_clusters * 2) - 2
        else:
            self.__num_of_kmeans_clusters = len(self.kmeans_cluster_node_id_to_cluster)
        self.__max_hierarchical_cluster_node_id = (self.__num_of_kmeans_clusters * 2) - 2
        self.hierarchical_cluster_node_id_to_cluster = {}
        self.taxonomy_label_to_color = self.__get_taxonomy_label_to_color(color_df)
        self.tree = None
        

    def add_labels(self, cluster_node_id_to_label):
        for cluster_node_id, cluster in self.kmeans_cluster_node_id_to_cluster.items():
            cluster.taxonomy_label = cluster_node_id_to_label[cluster_node_id]  
            for subreddit in cluster.children:
                subreddit.taxonomy_label = cluster_node_id_to_label[cluster_node_id]
        for cluster_node_id, cluster in self.hierarchical_cluster_node_id_to_cluster.items():
            cluster.taxonomy_label = cluster_node_id_to_label[cluster_node_id]   

    def get_data(self, include_subreddits):
        return self.__get_data_helper(self.tree, include_subreddits)
    
    def reduce_tree(self):
        self.__helper_reduce_tree(self.tree)

    def get_sorted_distances(self):
        cluster_to_sorted_clusters = {}
        subreddit_to_sorted_subreddits = {}
        self.__get_centers(self.tree, cluster_to_sorted_clusters, subreddit_to_sorted_subreddits)
        for cluster in cluster_to_sorted_clusters.keys():
            for compare_cluster in cluster_to_sorted_clusters.keys():
                if cluster != compare_cluster and compare_cluster not in cluster_to_sorted_clusters[cluster] and cluster not in cluster_to_sorted_clusters[compare_cluster]:
                    cluster_to_sorted_clusters[cluster][compare_cluster] = np.linalg.norm(cluster.center - compare_cluster.center)
                    cluster_to_sorted_clusters[compare_cluster][cluster] = np.linalg.norm(cluster.center - compare_cluster.center)
        for cluster, compare_cluster_to_distance in cluster_to_sorted_clusters.items():
            cluster.nearest_neighbors = sorted(compare_cluster_to_distance.items(), key=lambda x:x[1])
               
    def __get_centers(self, tree, cluster_to_sorted_clusters, subreddit_to_sorted_subreddits):
        has_subreddit_children = False
        if type(tree) == Cluster:
            children = tree.children
            for child in children:
                if type(child) == Subreddit:
                    has_subreddit_children = True
                self.__get_centers(child, cluster_to_sorted_clusters, subreddit_to_sorted_subreddits)
        else:
            subreddit_to_sorted_subreddits[tree] = {}
        if has_subreddit_children:
            cluster_to_sorted_clusters[tree] = {}
        
    def __helper_reduce_tree(self, tree):
        if type(tree) is Cluster:
            children = []
            total_tsne_x = 0
            total_tsne_y = 0
            tree.subreddit_count = 0
            tree.comment_count = 0
            children_taxonomy_label_to_children = {}
            for i in range(len(tree.children)): 
                child = tree.children[i]
                if type(child) == Cluster and len(child.children) == 1:
                    child = child.children[0]
                    tree.children[i] = child
                
                self.__helper_reduce_tree(child)
                if type(child) is Cluster and (tree.taxonomy_label == child.taxonomy_label or (type(child.taxonomy_label) is float and math.isnan(child.taxonomy_label))):
                    has_subreddit_grandchild = False
                    for grandchild in child.children:
                        children.append(grandchild)
                        if type(grandchild) is Cluster:
                            tree.subreddit_count += grandchild.subreddit_count
                            if grandchild.taxonomy_label not in children_taxonomy_label_to_children:
                                children_taxonomy_label_to_children[grandchild.taxonomy_label] = []
                            children_taxonomy_label_to_children[grandchild.taxonomy_label].append(grandchild)                                
                        else:
                            tree.subreddit_count += 1
                            has_subreddit_grandchild = True
                    if has_subreddit_grandchild:
                        if len(tree.center) == 0:
                            tree.center = child.center
                        else:
                            tree.center = (tree.center + child.center) / 2

                else:
                    children.append(child)
                    if type(child) is Cluster:
                        tree.subreddit_count += child.subreddit_count
                        if child.taxonomy_label not in children_taxonomy_label_to_children:
                            children_taxonomy_label_to_children[child.taxonomy_label] = []
                        children_taxonomy_label_to_children[child.taxonomy_label].append(child)
                    else:
                        tree.subreddit_count += 1
                
                tree.comment_count += child.comment_count
                total_tsne_x += child.tsne_x
                total_tsne_y += child.tsne_y
            for children_taxonomy_label, repeat_children in children_taxonomy_label_to_children.items():
                if len(repeat_children) > 1:
                    new_child = repeat_children[0]
                    for repeat_child in repeat_children[1:]:
                        children.remove(repeat_child)
                        new_child.center = (new_child.center + repeat_child.center) / 2
                        new_child.children.extend(repeat_child.children)
                        new_child.tsne_x += repeat_child.tsne_x
                        new_child.tsne_y += repeat_child.tsne_y
                        new_child.comment_count += repeat_child.comment_count
                        new_child.subreddit_count += repeat_child.subreddit_count
                    new_child.tsne_x /= len(repeat_children)
                    new_child.tsne_y /= len(repeat_children)
            updated_level = self.__calculate_level(children)
            if tree.level != updated_level:
                self.level_to_clusters[tree.level].remove(tree)
                if len(self.level_to_clusters[tree.level]) == 0:
                    self.level_to_clusters.pop(tree.level)
                tree.level = updated_level
                if updated_level not in self.level_to_clusters:
                    self.level_to_clusters[updated_level] = []
                self.level_to_clusters[updated_level].append(tree)
            if len(children) > 0:
                total_tsne_x /= len(children)
                total_tsne_y /= len(children)
            tree.tsne_x = total_tsne_x
            tree.tsne_y = total_tsne_y
            tree.children = children
    
    def __get_subreddits(self, clusters_df, tsne_df, subreddit_comment_counts_df, subreddit_centers, month):
        subreddits = []
        for index, row in clusters_df.iterrows():
            tsne_x = tsne_df["tsne_1"][index]
            tsne_y = tsne_df["tsne_2"][index]
            comment_count = subreddit_comment_counts_df["count"][index]
            subreddit = Subreddit(int(row[month + "_kmeans_clusters"]), index, row["subreddit"], comment_count, tsne_x, tsne_y)
            subreddit.center = subreddit_centers[index]
            subreddits.append(subreddit)
        return subreddits
    
    def __get_kmeans_cluster_node_id_to_subreddits(self, kmeans_cluster_node_ids, subreddits, cluster_centers):
        kmeans_cluster_node_id_to_cluster = {}
        for i in range(len(kmeans_cluster_node_ids)):
            kmeans_cluster_node_id = kmeans_cluster_node_ids[i]
            subreddit = subreddits[i]

            if kmeans_cluster_node_id not in kmeans_cluster_node_id_to_cluster:
                kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id] = Cluster(kmeans_cluster_node_id, cluster_centers[kmeans_cluster_node_id])
                kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level = 1
                if kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level not in self.level_to_clusters:
                    self.level_to_clusters[kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level] = []
                self.level_to_clusters[kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level].append(kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id])
            kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].add_children(subreddit)
        return kmeans_cluster_node_id_to_cluster
    
    def get_tree(self, parent_cluster_id_to_children_cluster_ids):
        self.tree = self.__get_hierarchical_cluster_node_id_to_cluster(self.__max_hierarchical_cluster_node_id, self.hierarchical_cluster_node_id_to_cluster, parent_cluster_id_to_children_cluster_ids)

    def __get_hierarchical_cluster_node_id_to_cluster(
        self, cluster_node_id, hierarchical_cluster_node_id_to_cluster, parent_cluster_id_to_children_cluster_ids
    ):
        if cluster_node_id < self.__num_of_kmeans_clusters:
            return self.kmeans_cluster_node_id_to_cluster[cluster_node_id]
        else:
            siblings = parent_cluster_id_to_children_cluster_ids[cluster_node_id]
            cluster = Cluster(cluster_node_id, [])
            children = []
            for sibling in siblings:
                child = self.__get_hierarchical_cluster_node_id_to_cluster(sibling, hierarchical_cluster_node_id_to_cluster, parent_cluster_id_to_children_cluster_ids)
                children.append(child)
                cluster.add_children(child)
            cluster.level = self.__calculate_level(children)
            hierarchical_cluster_node_id_to_cluster[cluster_node_id] = cluster
            if hierarchical_cluster_node_id_to_cluster[cluster_node_id].level not in self.level_to_clusters:
                self.level_to_clusters[hierarchical_cluster_node_id_to_cluster[cluster_node_id].level] = []
            self.level_to_clusters[hierarchical_cluster_node_id_to_cluster[cluster_node_id].level].append(hierarchical_cluster_node_id_to_cluster[cluster_node_id])
            return cluster
        
    def __calculate_level(self, children):
        if len(children) == 0:
            return 1
        return max(list(map(self.__get_level, children))) + 1
    
    def __get_taxonomy_label_to_color(self, color_df):
        taxonomy_label_to_color = {}
        for index, row in color_df.iterrows():
            taxonomy_label_to_color[row["label"]] = row["color"]
        taxonomy_label_to_color[""] = BASE_COLOR
        return taxonomy_label_to_color
        
    def __get_data_helper(self, tree, include_subreddits):
        taxonomy_label = ""
        
        color = self.taxonomy_label_to_color[""]
        if type(tree.taxonomy_label) == str:
            taxonomy_label = tree.taxonomy_label
            color = str(self.taxonomy_label_to_color[taxonomy_label.split("|")[0]])
        nearest_neighbors = []
        if len(tree.nearest_neighbors) > 0:
            for i in range(0, 10):
                nearest_neighbors.append(str(tree.nearest_neighbors[i][0].node_id))
            nearest_neighbors = ",".join(nearest_neighbors)
        if type(tree) is Cluster:
            children = []
            for child in tree.children:
                if type(child) == Cluster or include_subreddits:
                    children.append(self.__get_data_helper(child, include_subreddits))
            top_subreddits = ",".join(list(map(self.__get_subreddit_label, tree.get_top_subreddits("comment_count")[0:10])))
            parent_node_id = ""
            return {"node_id": str(tree.node_id), "comment_count": str(tree.comment_count), 
                "level": str(tree.level), "taxonomy_label": taxonomy_label, 
                "tsne_x": str(tree.tsne_x), "tsne_y": str(tree.tsne_y), "children": children, "subreddit_count": str(tree.subreddit_count), 
                "top_subreddits_by_comment": top_subreddits, "subreddit_count": str(tree.subreddit_count), 
                "color": color, "nearest_neighbors": nearest_neighbors}

        return {"node_id": str(tree.node_id) + "_" + str(tree.index), "cluster_label": str(tree.node_id), 
                "comment_count": str(tree.comment_count), 
                "level": str(tree.level), "taxonomy_label": taxonomy_label, 
                "tsne_x": str(tree.tsne_x), "tsne_y": str(tree.tsne_y), "subreddit": tree.subreddit_label, "subreddit_count": "1", 
                "color": color, "nearest_neighbors": nearest_neighbors}

    def test_test(self, j, ktov):
        if "children" not in j:
            j["nearest_neighbors"] = ktov[j["subreddit"]]
            return 1
        else:
            for c in j["children"]:
                self.test_test(c, ktov)
            return 2
    
    def __get_subreddit_label(self, subreddit):
        return subreddit.subreddit_label
    
    def __get_level(self, cluster):
        return cluster.level

class ClusterAndSubredditData:
    def __init__(self, node_id, comment_count, tsne_x, tsne_y):
        self.node_id = node_id
        self.comment_count = comment_count
        self.__nearest_neighbors = []
        self.__tsne_x = tsne_x
        self.__tsne_y = tsne_y
        self.__parent = None
        self.__level = -1
        self.__taxonomy_label = None

    @property
    def parent(self):
        return self.__parent
    
    @parent.setter
    def parent(self, p):
        self.__parent = p

    @property
    def level(self):
        return self.__level
    
    @level.setter
    def level(self, l):
        self.__level = l

    @property
    def nearest_neighbors(self):
        return self.__nearest_neighbors
    
    @nearest_neighbors.setter
    def nearest_neighbors(self, nn):
        self.__nearest_neighbors = nn

    @property
    def taxonomy_label(self):
        return self.__taxonomy_label
    
    @taxonomy_label.setter
    def taxonomy_label(self, label):
        self.__taxonomy_label = label 

    @property
    def tsne_x(self):
        return self.__tsne_x
    
    @tsne_x.setter
    def tsne_x(self, x):
        self.__tsne_x = x

    @property
    def tsne_y(self):
        return self.__tsne_y
    
    @tsne_y.setter
    def tsne_y(self, y):
        self.__tsne_y = y


class Cluster(ClusterAndSubredditData):
    def __init__(self, node_id, center):
        ClusterAndSubredditData.__init__(self, node_id, 0, 0, 0)
        self.__children = []
        self.subreddit_count = 0
        self.subreddits = set()
        self.center = center
    
    @property
    def children(self):
        return self.__children
    
    @children.setter
    def children(self, c):
        self.__children = c

    def add_children(self, child):
        if type(child) is Subreddit:
            self.subreddits.add(child)
        else:
            self.subreddits = self.subreddits.union(child.subreddits)
        self.__children.append(child)
        self.subreddit_count = len(self.subreddits)
        self.comment_count += child.comment_count

    def get_top_subreddits(self, count_type):
        subreddit_to_count = {}
        for subreddit in self.subreddits:
            if count_type == "comment_count":
                count = subreddit.comment_count
            subreddit_to_count[subreddit] = count
        subreddit_to_count = {subreddit: count for subreddit, count in sorted(subreddit_to_count.items(), key=lambda item: item[1], reverse=True)}
        top_subreddits = []
        if count_type == "comment_count":
            for subreddit, count in subreddit_to_count.items():
                top_subreddits.append(subreddit)
        return top_subreddits

class Subreddit(ClusterAndSubredditData):
    def __init__(self, node_id, index, subreddit_label, comment_count, tsne_x, tsne_y):
        ClusterAndSubredditData.__init__(self, node_id, comment_count, tsne_x, tsne_y)
        self.subreddit_label = subreddit_label
        self.index = index
        self.level = 0

class GenerateMapping:
    def __init__(self, orig_tree_obj, tree_obj, human_taxonomy_df):
        self.__new_cluster = orig_tree_obj.kmeans_cluster_node_id_to_cluster[NEW_CLUSTER_ID]
        self.intersect = set(map(self.__get_subreddit_label, orig_tree_obj.subreddits)).intersection(set(map(self.__get_subreddit_label, tree_obj.subreddits)))
        self.level_1_to_2 = {}
        self.orig_parent_to_cluster_node_ids = {}
        self.__num_of_kmeans_clusters = len(orig_tree_obj.kmeans_cluster_node_id_to_cluster) - 1
        self.__max_hierarchical_cluster_node_id = (self.__num_of_kmeans_clusters * 2) - 2
        self.__get_base_clusters(human_taxonomy_df)
        self.orig_child_to_parents = self.__get_child_to_parents(orig_tree_obj)
        self.parent_cluster_node_id_to_cluster_node_ids = {}
        self.cluster_node_id_to_label = {}
        self.cluster_node_id_to_orig_taxonomy_label = self.__get_matches(tree_obj, orig_tree_obj)
        
        

    def __get_child_to_parents(self, tree_obj):
        child_to_parents = {}
        self.__get_child_to_parents_helper(tree_obj.tree, child_to_parents)
        # add Porn
        for kmeans_cluster_node_id, cluster in tree_obj.kmeans_cluster_node_id_to_cluster.items():
            if cluster.taxonomy_label == "Porn" and cluster.node_id != NSFW_ID:
                child_to_parents[cluster] = [tree_obj.hierarchical_cluster_node_id_to_cluster[NSFW_ID]]
            if kmeans_cluster_node_id == NEW_CLUSTER_ID:
                child_to_parents[cluster] = [tree_obj.hierarchical_cluster_node_id_to_cluster[ROOT_ID]]
        return child_to_parents
        

    def __get_child_to_parents_helper(self, tree, child_to_parents):
        for child in tree.children:
            if type(child) == Cluster:
                if child not in child_to_parents:
                    child_to_parents[child] = []
                child_to_parents[child].append(tree)
                self.__get_child_to_parents_helper(child, child_to_parents)
            
    def __get_base_clusters(self, human_taxonomy_df):
        for index, row in human_taxonomy_df.iterrows():
            if row["cluster_node_id"] < self.__num_of_kmeans_clusters:
                if row["cluster_node_id"] not in self.level_1_to_2:
                    self.level_1_to_2[row["cluster_node_id"]] = []
                self.level_1_to_2[row["cluster_node_id"]].append(row["parent_cluster_node_id"])
                self.orig_parent_to_cluster_node_ids[row["parent_cluster_node_id"]] = []

    def __get_initial_matches_curr_to_orig(self, tree_obj, orig_tree_obj):
        curr_to_orig = {}
        orig_to_cur = {}
        set_of_labels = set()
        for kmeans_cluster_node_id, cluster in tree_obj.kmeans_cluster_node_id_to_cluster.items():
            max_kmeans_cluster_node_id = None
            max_sim = 0
            curr_to_orig[cluster] = {}
            threshold = SIMILARITY_THRESHOLD
            reduce = 0
            while len(curr_to_orig[cluster]) == 0 and threshold > 0:
                for orig_kmeans_cluster_node_id, orig_cluster in orig_tree_obj.kmeans_cluster_node_id_to_cluster.items():
                    sim = self.__get_similarity(orig_cluster, cluster)
                    if sim >= threshold:
                        curr_to_orig[cluster][orig_cluster] = sim
                        if orig_cluster not in orig_to_cur:
                            orig_to_cur[orig_cluster] = {}
                        orig_to_cur[orig_cluster][cluster] = sim
                        set_of_labels.add(orig_cluster)
                reduce += SIMILARITY_THRESHOLD_REDUCE
                threshold -= reduce
            if len(curr_to_orig[cluster]) == 0:
                curr_to_orig[cluster][self.__new_cluster] = 0
                if self.__new_cluster not in orig_to_cur:
                    orig_to_cur[self.__new_cluster] = {}
                orig_to_cur[self.__new_cluster][cluster] = 0
        return curr_to_orig, orig_to_cur, set_of_labels

    def __get_initial_matches_orig_to_curr(self, tree_obj, orig_tree_obj, curr_to_orig, orig_to_cur, set_of_labels):
        for orig_kmeans_cluster_node_id, orig_cluster in orig_tree_obj.kmeans_cluster_node_id_to_cluster.items():
            max_kmeans_cluster_node_id = None
            max_sim = 0
            if orig_cluster not in set_of_labels:
                for kmeans_cluster_node_id, cluster in tree_obj.kmeans_cluster_node_id_to_cluster.items():
                    sim = self.__get_similarity(orig_cluster, cluster)
                    if sim >= max_sim:
                        max_sim = sim
                        max_kmeans_cluster_node_id = cluster
                if max_kmeans_cluster_node_id not in curr_to_orig:
                    curr_to_orig[max_kmeans_cluster_node_id] = {}
                curr_to_orig[max_kmeans_cluster_node_id][orig_cluster] = max_sim
                if orig_cluster not in orig_to_cur:
                    orig_to_cur[orig_cluster] = {}
                orig_to_cur[orig_cluster][cluster] = max_sim

    def __match_clusters(self, curr_to_orig, orig_to_cur, matches_curr_to_orig, matches_orig_to_curr, matching_iter):
        for cluster, orig_clusters in curr_to_orig.items():
            max_orig_taxonomy_label = max(orig_clusters, key=orig_clusters.get)
            if matching_iter == 0 and ((cluster not in matches_curr_to_orig and max_orig_taxonomy_label not in matches_orig_to_curr) or max_orig_taxonomy_label.node_id == "Porn"):
                max_orig_taxonomy_label = max(orig_clusters, key=orig_clusters.get)
                if max(orig_to_cur[max_orig_taxonomy_label], key=orig_to_cur[max_orig_taxonomy_label].get) == cluster or max_orig_taxonomy_label.node_id == "Porn":
                    if cluster not in matches_curr_to_orig:
                        matches_curr_to_orig[cluster] = []
                    matches_curr_to_orig[cluster].append(max_orig_taxonomy_label)
                    if max_orig_taxonomy_label not in matches_orig_to_curr:
                        matches_orig_to_curr[max_orig_taxonomy_label] = []
                    matches_orig_to_curr[max_orig_taxonomy_label].append(cluster)
            if matching_iter == 1 and cluster not in matches_curr_to_orig:
                unmatched_orig_taxonomy_labels = {}
                for orig_cluster, sim in orig_clusters.items():
                    if orig_cluster not in matches_orig_to_curr:
                        unmatched_orig_taxonomy_labels[orig_cluster] = sim
                if len(unmatched_orig_taxonomy_labels) != 0:
                    max_orig_taxonomy_label = max(unmatched_orig_taxonomy_labels, key=unmatched_orig_taxonomy_labels.get)                        
                if cluster not in matches_curr_to_orig:
                    matches_curr_to_orig[cluster] = []
                matches_curr_to_orig[cluster].append(max_orig_taxonomy_label)
                if max_orig_taxonomy_label not in matches_orig_to_curr:
                    matches_orig_to_curr[max_orig_taxonomy_label] = []
                matches_orig_to_curr[max_orig_taxonomy_label].append(cluster)

    def __match_remaining_labels(self, orig_to_cur, matches_curr_to_orig, matches_orig_to_curr):
        for orig_cluster, clusters in orig_to_cur.items():
            if orig_cluster not in matches_orig_to_curr:
                for cluster, sim in clusters.items():
                    if sim > SIMILARITY_THRESHOLD:
                        matches_curr_to_orig[cluster].append(orig_cluster)
                        if orig_cluster not in matches_orig_to_curr:
                            matches_orig_to_curr[orig_cluster] = []
                        matches_orig_to_curr[orig_cluster].append(cluster)

    def __get_cluster_node_id_to_parent_cluster_node_ids(self, matches_orig_to_curr):
        cluster_node_id_to_parent_cluster_node_ids = {}
        for child, parents in self.orig_child_to_parents.items():
            for parent in parents:
                if child in matches_orig_to_curr:
                    for i in range(len(matches_orig_to_curr[child])):
                        match = matches_orig_to_curr[child][i]
                        if parent.node_id not in self.parent_cluster_node_id_to_cluster_node_ids:
                            self.parent_cluster_node_id_to_cluster_node_ids[parent.node_id] = []
                        if match.node_id not in cluster_node_id_to_parent_cluster_node_ids:
                            cluster_node_id_to_parent_cluster_node_ids[match.node_id] = []
                        self.parent_cluster_node_id_to_cluster_node_ids[parent.node_id].append(match.node_id)
                        cluster_node_id_to_parent_cluster_node_ids[match.node_id].append(parent.node_id)
                        self.cluster_node_id_to_label[parent.node_id] = parent.taxonomy_label
                        if match.node_id not in self.cluster_node_id_to_label:
                            self.cluster_node_id_to_label[match.node_id] = ""
                        if child.taxonomy_label not in self.cluster_node_id_to_label[match.node_id]:
                            self.cluster_node_id_to_label[match.node_id] += child.taxonomy_label + "|"
           
                else:
                    if parent.node_id not in self.parent_cluster_node_id_to_cluster_node_ids:
                        self.parent_cluster_node_id_to_cluster_node_ids[parent.node_id] = []
                    if child.node_id not in cluster_node_id_to_parent_cluster_node_ids:
                            cluster_node_id_to_parent_cluster_node_ids[child.node_id] = []
                    if child.node_id != NEW_CLUSTER_ID:
                        self.parent_cluster_node_id_to_cluster_node_ids[parent.node_id].append(child.node_id)
                        cluster_node_id_to_parent_cluster_node_ids[child.node_id].append(parent.node_id)
                        if type(parent.taxonomy_label) == str:
                            self.cluster_node_id_to_label[parent.node_id] = parent.taxonomy_label + "|"
                        else:
                            self.cluster_node_id_to_label[parent.node_id] = parent.taxonomy_label

                        if type(child.taxonomy_label) == str:
                            self.cluster_node_id_to_label[child.node_id] = child.taxonomy_label + "|"
                        else:
                            self.cluster_node_id_to_label[child.node_id] = child.taxonomy_label
        return cluster_node_id_to_parent_cluster_node_ids

    def __eliminate_unwanted_labels(self):
        for cluster_node_id, label in self.cluster_node_id_to_label.items():
            if cluster_node_id != self.__max_hierarchical_cluster_node_id:
                all_labels = label.split("|")
                if len(all_labels) > 2 and "Porn" in all_labels:
                    new_all_labels = ""
                    for l in all_labels:
                        if len(l) > 0 and l != "Porn":
                            new_all_labels += l + "|"
                    label = new_all_labels
                if label[-1] == "|":
                    self.cluster_node_id_to_label[cluster_node_id] = label[:-1]

    def __get_multi_label_cluster_node_ids(self, cluster_node_id_to_parent_cluster_node_ids):
        multi_label_cluster_node_ids = []
        for cluster_node_id, labels in self.cluster_node_id_to_label.items():
            if cluster_node_id != self.__max_hierarchical_cluster_node_id and len(labels.split("|")) > 1 and len(set(cluster_node_id_to_parent_cluster_node_ids[cluster_node_id])) > 1:
                multi_label_cluster_node_ids.append(cluster_node_id)
        return multi_label_cluster_node_ids

    def __get_multi_label_cluster_node_id_to_location(self, multi_label_cluster_node_ids, cluster_node_id_to_parent_cluster_node_ids):
        multi_label_cluster_node_id_to_location = {}
        for multi_label_cluster_node_id in multi_label_cluster_node_ids:
            parents = set(cluster_node_id_to_parent_cluster_node_ids[multi_label_cluster_node_id])
            common_parent = self.__find_common_parent(cluster_node_id_to_parent_cluster_node_ids, parents)
            if common_parent != self.__max_hierarchical_cluster_node_id:
                multi_label_cluster_node_id_to_location[multi_label_cluster_node_id] = common_parent
                if multi_label_cluster_node_id not in self.parent_cluster_node_id_to_cluster_node_ids[common_parent]:
                    self.parent_cluster_node_id_to_cluster_node_ids[common_parent].append(multi_label_cluster_node_id)
        return multi_label_cluster_node_id_to_location
    def __update_parent_cluster_node_id_to_cluster_node_ids(self, multi_label_cluster_node_id_to_location):
        empty_cluster_node_ids = set()
        for parent_cluster_node_id, cluster_node_ids in self.parent_cluster_node_id_to_cluster_node_ids.copy().items():
            new_cluster_node_ids = []
            for cluster_node_id in cluster_node_ids:
                if (cluster_node_id in multi_label_cluster_node_id_to_location and parent_cluster_node_id == multi_label_cluster_node_id_to_location[cluster_node_id]) or (cluster_node_id not in multi_label_cluster_node_id_to_location and cluster_node_id not in new_cluster_node_ids):
                    new_cluster_node_ids.append(cluster_node_id)
            if len(new_cluster_node_ids) == 0:
                empty_cluster_node_ids.add(parent_cluster_node_id)
                self.parent_cluster_node_id_to_cluster_node_ids.pop(parent_cluster_node_id)
                for parent in cluster_node_id_to_parent_cluster_node_ids[parent_cluster_node_id]:
                    self.parent_cluster_node_id_to_cluster_node_ids[parent].remove(parent_cluster_node_id)
            else:
                self.parent_cluster_node_id_to_cluster_node_ids[parent_cluster_node_id] = new_cluster_node_ids
        for parent_cluster_node_id, cluster_node_ids in self.parent_cluster_node_id_to_cluster_node_ids.copy().items():
            new_cluster_node_ids = []
            for cluster_node_id in cluster_node_ids.copy():
                if cluster_node_id not in empty_cluster_node_ids:
                    new_cluster_node_ids.append(cluster_node_id)
                if parent_cluster_node_id == 138 and self.cluster_node_id_to_label[cluster_node_id] != "Porn":
                    self.parent_cluster_node_id_to_cluster_node_ids[parent_cluster_node_id].remove(cluster_node_id)


    def __get_matches(self, tree_obj, orig_tree_obj):
        matches_curr_to_orig = {}
        matches_orig_to_curr = {}
        orig_cluster_node_id_to_cluster_node_id = {}
        orig_level_1_cluster_node_id_to_cluster_node_ids = {}
        curr_to_orig, orig_to_cur, set_of_labels = self.__get_initial_matches(tree_obj, orig_tree_obj)
        self.__get_initial_matches_orig_to_curr(tree_obj, orig_tree_obj, curr_to_orig, orig_to_cur, set_of_labels)
        
        matching_iter = 0
        while (len(matches_curr_to_orig) != self.__num_of_kmeans_clusters or  len(matches_orig_to_curr) != self.__num_of_kmeans_clusters) and matching_iter != 3:
            if matching_iter == 0 or matching_iter == 1:
                # ensures that all clusters for the current month have one label
                self.__match_clusters(curr_to_orig, orig_to_cur, matches_curr_to_orig, matches_orig_to_curr, matching_iter)
            if matching_iter == 2:
                # ensures that all potential labels are used
                self.__match_remaining_labels(orig_to_cur, matches_curr_to_orig, matches_orig_to_curr)
            matching_iter += 1

        cluster_node_id_to_parent_cluster_node_ids = self.__get_cluster_node_id_to_parent_cluster_node_ids(matches_orig_to_curr)
        self.__eliminate_unwanted_labels()
        multi_label_cluster_node_ids = self.__get_multi_label_cluster_node_ids(cluster_node_id_to_parent_cluster_node_ids)
        multi_label_cluster_node_id_to_location = self.__get_multi_label_cluster_node_id_to_location(self, multi_label_cluster_node_ids, cluster_node_id_to_parent_cluster_node_ids)
        self.__update_parent_cluster_node_id_to_cluster_node_ids(multi_label_cluster_node_id_to_location)
       

    def __find_common_parent(self, cluster_node_id_to_parent_cluster_node_ids, cluster_node_ids):
        family_trees = []
        for cluster_node_id in cluster_node_ids:
            family_tree = [cluster_node_id]
            self.__get_family_trees(cluster_node_id_to_parent_cluster_node_ids, cluster_node_id, family_tree, family_trees)
        curr = family_trees[0]
        for family_tree in family_trees[1:]:
            curr = [ e for e in family_tree if e in curr] 
        return curr[0]
    def __get_family_trees(self, cluster_node_id_to_parent_cluster_node_ids, cluster_node_id, family_tree, family_trees):
        if self.__max_hierarchical_cluster_node_id in family_tree:
            family_trees.append(family_tree)
        elif len(cluster_node_id_to_parent_cluster_node_ids[cluster_node_id]) == 1:
            family_tree.append(cluster_node_id_to_parent_cluster_node_ids[cluster_node_id][0])
            self.__get_family_trees(cluster_node_id_to_parent_cluster_node_ids, cluster_node_id_to_parent_cluster_node_ids[cluster_node_id][0], family_tree, family_trees)
        else:
            for parent in cluster_node_id_to_parent_cluster_node_ids[cluster_node_id]:
                new_family_tree = family_tree.copy()
                new_family_tree.append(parent)
                self.__get_family_trees(cluster_node_id_to_parent_cluster_node_ids, parent, new_family_tree, family_trees)
             

    def __get_similarity(self, orig_cluster, cluster):
            orig_cluster_subreddit_labels = set(map(self.__get_subreddit_label, list(filter(lambda subreddit: self.__get_subreddit_label(subreddit) in self.intersect, orig_cluster.subreddits))))
            cluster_subreddit_labels = set(map(self.__get_subreddit_label, list(filter(lambda subreddit: self.__get_subreddit_label(subreddit) in self.intersect, cluster.subreddits))))
            if len(orig_cluster_subreddit_labels
                    .intersection(cluster_subreddit_labels)) == 0:
                return 0
            return len(orig_cluster_subreddit_labels
                    .intersection(cluster_subreddit_labels)) / len(orig_cluster_subreddit_labels
                                                                        .union(cluster_subreddit_labels))

    def __get_intersection(self, orig_cluster, cluster):
            orig_cluster_subreddit_labels = set(map(self.__get_subreddit_label, orig_cluster.subreddits))
            cluster_subreddit_labels = set(map(self.__get_subreddit_label, cluster.subreddits))
            return orig_cluster_subreddit_labels.intersection(cluster_subreddit_labels) 
    
    def __get_subreddit_label(self, subreddit):
        return subreddit.subreddit_label

def get_subreddit_label(subreddit):
    return subreddit.subreddit_label

def main():
    parser = argparse.ArgumentParser(
                    prog='Get Reddit Map Data' ,
                    description='To retreive the JSONs used by the Reddit Map visualizations, this program processes the KMeans clusters for each month of data given by the user.')
    parser.add_argument('-n', '--month_name', help="year-month format for time period the current data is fro,")
    parser.add_argument('-m', '--sklearn_cluster_model_path', help="path to the kmeans model trained on the current data")
    parser.add_argument('-k', '--keyed_vectors_path', help="path to the keyed vectors representing each kmeans cluster")
    parser.add_argument('-c', '--clusters_path', help="path to a csv with the following columns: 'subreddit', month_name + '_kmeans_clusters'. The subreddit column contains the subreddit name and the kmeans_clusters column represents the id of the cluster the subreddit belongs to")
    parser.add_argument('-t', '--tsne_path', help="path to a csv with the following columns: 'subreddit', 'tsne_1', 'tsne_2'. Each row represnts the tsne coordinates of the given subreddit")
    parser.add_argument('-s', '--subreddit_counts_path', help="path to csv with the following columns: 'subreddit', 'count'. Each row represents the number of comments posted on the given subreddit for the current month of data")

    args = parser.parse_args()
    current_month = args.month_name

    # retrieve the base model's JSON to help with mapping human-generated labels onto the current_month's worth of data
    orig_kmeans_model_path = '../data/kmeans_data/RC_2021-04_kmeans/sklearn_cluster_model.joblib'
    orig_keyed_vectors_path = '../data/community2vec/RC_2021-04/best_model/keyedVectors'
    labels_path = '../data/mapping_data/human_taxonomy.json'
    orig_clusters_path = '../data/mapping_data/clusters.csv'
    orig_tsne_path = '../data/mapping_data/tsne.csv'
    orig_subreddit_comment_counts_path = '../data/mapping_data/subreddit_counts.csv'
    orig_parent_path = '../data/mapping_data/parents.csv'
    orig_human_taxonomy_path = '../data/mapping_data/human_taxonomy_revised.csv'
    orig_parent_df = pd.read_csv(orig_parent_path)
    orig_human_taxonomy_df = pd.read_csv(orig_human_taxonomy_path)
    orig_clusters_df = pd.read_csv(orig_clusters_path)
    orig_tsne_df = pd.read_csv(orig_tsne_path)
    orig_subreddit_comment_counts_df = pd.read_csv(orig_subreddit_comment_counts_path)
    test = gm.KeyedVectors.load(orig_keyed_vectors_path)
    
    orig_kmeans_cluster_centers = joblib.load(orig_kmeans_model_path).cluster_centers_
    orig_subreddit_centers = gm.KeyedVectors.load(orig_keyed_vectors_path).get_normed_vectors()
    orig_human_taxonomy_rules_mapping_path = "../data/mapping_data/hierarchy_labels_rules_mapping.csv"
    orig_human_taxonomy_rules_mapping_df = pd.read_csv(orig_human_taxonomy_rules_mapping_path)
    orig_parent_cluster_node_id_to_cluster_node_id_rules_mapping = {}
    orig_cluster_node_id_to_label_rules_mapping = {}
    for index, row in orig_human_taxonomy_rules_mapping_df.iterrows():
        if row["parent_cluster_node_id"] not in orig_parent_cluster_node_id_to_cluster_node_id_rules_mapping:
            orig_parent_cluster_node_id_to_cluster_node_id_rules_mapping[row["parent_cluster_node_id"]] = []
        orig_parent_cluster_node_id_to_cluster_node_id_rules_mapping[row["parent_cluster_node_id"]].append(row["cluster_node_id"])
    for index, row in orig_human_taxonomy_rules_mapping_df.iterrows():
        orig_cluster_node_id_to_label_rules_mapping[row["cluster_node_id"]] = row["label"]
    orig_tree_obj_rules_mapping = Tree(orig_clusters_df, orig_tsne_df, orig_subreddit_comment_counts_df, orig_human_taxonomy_rules_mapping_df, orig_kmeans_cluster_centers, orig_subreddit_centers, BASE_MONTH)
    orig_tree_obj_rules_mapping.get_tree(orig_parent_cluster_node_id_to_cluster_node_id_rules_mapping)
    orig_tree_obj_rules_mapping.add_labels(orig_cluster_node_id_to_label_rules_mapping)
    orig_tree_obj_rules_mapping.reduce_tree()
    orig_tree_obj_rules_mapping.get_sorted_distances()

    # generate the JSON of the current_month' data
    kmeans_model_path = args.sklearn_cluster_model_path
    keyed_vectors_path = args.keyed_vectors_path
    clusters_path = args.clusters_path
    tsne_path = args.tsne_path
    subreddit_comment_counts_path = args.subreddit_counts_path
    kmeans_cluster_centers = joblib.load(kmeans_model_path).cluster_centers_
    subreddit_centers = gm.KeyedVectors.load(keyed_vectors_path).get_normed_vectors()
    clusters_df = pd.read_csv(clusters_path)
    tsne_df = pd.read_csv(tsne_path)
    subreddit_comment_counts_df = pd.read_csv(subreddit_comment_counts_path)
    tree_obj_rules_mapping = Tree(clusters_df, tsne_df, subreddit_comment_counts_df, orig_human_taxonomy_rules_mapping_df, kmeans_cluster_centers, subreddit_centers, current_month[3:])

    mapping = GenerateMapping(orig_tree_obj_rules_mapping, tree_obj_rules_mapping, orig_human_taxonomy_rules_mapping_df)
    tree_obj_rules_mapping.get_tree(mapping.parent_cluster_node_id_to_cluster_node_ids)
    tree_obj_rules_mapping.add_labels(mapping.cluster_node_id_to_label)
    tree_obj_rules_mapping.reduce_tree()
    tree_obj_rules_mapping.get_sorted_distances()
    tree_obj_rules_mapping.get_data(False)


if __name__ == "__main__":
    main()