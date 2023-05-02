import gensim.models as gm
import joblib
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import json
import math
import argparse
import os


class KMeansToAgglomerative:
    def __init__(self, kmeans_model_path):
        self.kmeans_model = joblib.load(kmeans_model_path)
        self.agglomerative_clustering_data = self.__agglomerative_clustering()

    def __get_flat_distance_matrix(self):
        cluster_centers = self.kmeans_model.cluster_centers_
        distance_matrix = np.zeros((len(cluster_centers), len(cluster_centers)))
        flat_distance_matrix = []
        start = 1
        for i in range(len(distance_matrix)):
            for j in range(start, len(distance_matrix[i])):
                distance_matrix[i][j] = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                flat_distance_matrix.append(distance_matrix[i][j])
            start += 1
        return flat_distance_matrix

    def __get_children_from_linkage_matrix(self, linkage_matrix):
        children = []
        for i in range(len(linkage_matrix[:, 0:2])):
            children.append([int(linkage_matrix[:, 0:2][i][0]), int(linkage_matrix[:, 0:2][i][1])])
        return children

    def __agglomerative_clustering(self):
        flat_distance_matrix = self.__get_flat_distance_matrix()
        linkage_matrix = hierarchy.linkage(flat_distance_matrix, "single")
        children = self.__get_children_from_linkage_matrix(linkage_matrix)
        return children

class NewTree:
    def __init__(self, clusters_df, tsne_df, subreddit_comment_counts_df, color_df, cluster_centers, subreddit_centers, month):
        self.level_to_clusters = {}
        self.subreddits = self.__get_subreddits(clusters_df, tsne_df, subreddit_comment_counts_df, subreddit_centers, month)
        self.kmeans_cluster_node_id_to_cluster = self.__get_kmeans_cluster_node_id_to_subreddits(clusters_df[month + "_kmeans_clusters"].to_numpy(), self.subreddits, cluster_centers)
        if month == "2021-04":
            self.kmeans_cluster_node_id_to_cluster[-1] = Cluster(-1, None)
            self.kmeans_cluster_node_id_to_cluster[-1].taxonomy_label = "New Cluster"
            self.kmeans_cluster_node_id_to_cluster[-1].level = 1
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
        """
        for subreddit in subreddit_to_sorted_subreddits.keys():
            for compare_subreddit in subreddit_to_sorted_subreddits.keys():
                if subreddit != compare_subreddit and compare_subreddit not in subreddit_to_sorted_subreddits[subreddit] and subreddit not in subreddit_to_sorted_subreddits[compare_subreddit]:
                    dist = np.linalg.norm(subreddit.center - compare_subreddit.center)
                    subreddit_to_sorted_subreddits[subreddit][compare_subreddit] = dist
                    subreddit_to_sorted_subreddits[compare_subreddit][subreddit] = dist
        """
        for cluster, compare_cluster_to_distance in cluster_to_sorted_clusters.items():
            cluster.nearest_neighbors = sorted(compare_cluster_to_distance.items(), key=lambda x:x[1])
        """
        for subreddit, compare_subreddit_to_distance in subreddit_to_sorted_subreddits.items():
            print("subreddit: ", subreddit.node_id, subreddit.index)
            subreddit.nearest_neighbors = sorted(compare_subreddit_to_distance.items(), key=lambda x:x[1])
        """
               
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
        taxonomy_label_to_color[""] = "#808080"
        return taxonomy_label_to_color
        
    def __get_data_helper(self, tree, include_subreddits):
        taxonomy_label = ""
        nearest_neighbors = []
        if len(tree.nearest_neighbors) > 0:
            for i in range(0, 10):
                nearest_neighbors.append(str(tree.nearest_neighbors[i][0].node_id))
        nearest_neighbors = ",".join(nearest_neighbors)
        color = self.taxonomy_label_to_color[""]
        if type(tree.taxonomy_label) == str:
            taxonomy_label = tree.taxonomy_label
            color = str(self.taxonomy_label_to_color[taxonomy_label.split("|")[0]])
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
    
    def __get_subreddit_label(self, subreddit):
        return subreddit.subreddit_label
    
    def __get_level(self, cluster):
        return cluster.level


class Tree:
    def __init__(self, clusters_df, tsne_df, subreddit_comment_counts_df, agglomerative_clustering_data, color_df):
        self.__num_of_kmeans_clusters = len(agglomerative_clustering_data) + 1
        self.__max_hierarchical_cluster_node_id = len(agglomerative_clustering_data) * 2
        self.level_to_clusters = {}
        subreddits = self.__get_subreddits(clusters_df, tsne_df, subreddit_comment_counts_df)
        self.kmeans_cluster_node_id_to_cluster = self.__get_kmeans_cluster_node_id_to_subreddits(clusters_df["kmeans"].to_numpy(), subreddits)
        self.hierarchical_cluster_node_id_to_cluster = {}
        self.tree = self.__get_hierarchical_cluster_node_id_to_cluster(
            self.__max_hierarchical_cluster_node_id, self.hierarchical_cluster_node_id_to_cluster, agglomerative_clustering_data
        )
        self.taxonomy_label_to_color = self.__get_taxonomy_label_to_color(color_df)

    def add_labels(self, cluster_node_id_to_label):
        for cluster_node_id, cluster in self.kmeans_cluster_node_id_to_cluster.items():
            cluster.taxonomy_label = cluster_node_id_to_label[cluster_node_id]  
            for subreddit in cluster.children:
                subreddit.taxonomy_label = cluster_node_id_to_label[cluster_node_id]
        for cluster_node_id, cluster in self.hierarchical_cluster_node_id_to_cluster.items():
            cluster.taxonomy_label = cluster_node_id_to_label[cluster_node_id]   

    def group_locals(self, root):
        if type(root) is Cluster and root.node_id == self.__max_hierarchical_cluster_node_id:
            local_children = []
            root_children = []
            total_tsne_x = 0
            total_tsne_y = 0
            subreddit_count = 0
            comment_count = 0
            local_cluster = Cluster(self.__max_hierarchical_cluster_node_id + 1)
            local_cluster.parent = root
            local_cluster.taxonomy_label = "Local"
            for child in root.children:
                if "local" in child.taxonomy_label.lower():
                    local_children.append(child)
                    subreddit_count += child.subreddit_count
                    comment_count += child.comment_count
                    total_tsne_x += child.tsne_x
                    total_tsne_y += child.tsne_y
                else:
                    root_children.append(child)
            local_cluster.level = self.__calculate_level(local_children)
            local_cluster.tsne_x = total_tsne_x / len(local_children)
            local_cluster.tsne_y = total_tsne_y / len(local_children)
            local_cluster.comment_count = comment_count
            local_cluster.subreddit_count = subreddit_count
            local_cluster.children = local_children
            root_children.append(local_cluster)
            root.children = root_children


    def reduce_tree(self, tree):
        if type(tree) is Cluster:
            children = []
            total_tsne_x = 0
            total_tsne_y = 0
            tree.subreddit_count = 0
            tree.comment_count = 0
            for child in tree.children: 
                self.reduce_tree(child)
                if type(child) is Cluster and (tree.taxonomy_label == child.taxonomy_label or (type(child.taxonomy_label) is float and math.isnan(child.taxonomy_label))):
                    child.parent = tree
                    for grandchild in child.children:
                        children.append(grandchild)
                        if type(grandchild) is Cluster:
                            tree.subreddit_count += grandchild.subreddit_count
                        else:
                            tree.subreddit_count += 1
                else:
                    children.append(child)
                    if type(child) is Cluster:
                            tree.subreddit_count += child.subreddit_count
                    else:
                        tree.subreddit_count += 1
                
                tree.comment_count += child.comment_count
                total_tsne_x += child.tsne_x
                total_tsne_y += child.tsne_y
            
            updated_level = self.__calculate_level(children)
            if tree.level != updated_level:
                self.level_to_clusters[tree.level].remove(tree)
                if len(self.level_to_clusters[tree.level]) == 0:
                    self.level_to_clusters.pop(tree.level)
                tree.level = updated_level
                if updated_level not in self.level_to_clusters:
                    self.level_to_clusters[updated_level] = []
                self.level_to_clusters[updated_level].append(tree)
            total_tsne_x /= len(children)
            total_tsne_y /= len(children)
            tree.tsne_x = total_tsne_x
            tree.tsne_y = total_tsne_y
            tree.children = children


    def __get_taxonomy_label_to_color(self, color_df):
        taxonomy_label_to_color = {}
        for index, row in color_df.iterrows():
            taxonomy_label_to_color[row["label"]] = row["color"]
        taxonomy_label_to_color[""] = "#808080"
        return taxonomy_label_to_color

    def __get_subreddits(self, clusters_df, tsne_df, subreddit_comment_counts_df):
        subreddits = []
        for index, row in clusters_df.iterrows():
            tsne_x = tsne_df["tsne_1"][index]
            tsne_y = tsne_df["tsne_2"][index]
            comment_count = subreddit_comment_counts_df["count"][index]
            subreddit = Subreddit(int(row["kmeans"]), index, row["subreddit"], comment_count, tsne_x, tsne_y)
            subreddits.append(subreddit)
        return subreddits

    def __get_kmeans_cluster_node_id_to_subreddits(self, kmeans_cluster_node_ids, subreddits):
        kmeans_cluster_node_id_to_cluster = {}
        for i in range(len(kmeans_cluster_node_ids)):
            kmeans_cluster_node_id = kmeans_cluster_node_ids[i]
            subreddit = subreddits[i]
            if kmeans_cluster_node_id not in kmeans_cluster_node_id_to_cluster:
                kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id] = Cluster(kmeans_cluster_node_id)
                kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level = 1
                if kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level not in self.level_to_clusters:
                    self.level_to_clusters[kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level] = []
                self.level_to_clusters[kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].level].append(kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id])
            subreddit.parent = kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id]
            kmeans_cluster_node_id_to_cluster[kmeans_cluster_node_id].add_children(subreddit)
        return kmeans_cluster_node_id_to_cluster
    
    def __get_hierarchical_cluster_node_id_to_cluster(
        self, cluster_node_id, hierarchical_cluster_node_id_to_cluster, agglomerative_clustering_data
    ):
        if cluster_node_id < self.__num_of_kmeans_clusters:
            return self.kmeans_cluster_node_id_to_cluster[cluster_node_id]
        else:
            siblings = agglomerative_clustering_data[cluster_node_id - self.__num_of_kmeans_clusters]
            child_1 = self.__get_hierarchical_cluster_node_id_to_cluster(siblings[0], hierarchical_cluster_node_id_to_cluster, agglomerative_clustering_data)
            child_2 = self.__get_hierarchical_cluster_node_id_to_cluster(siblings[1], hierarchical_cluster_node_id_to_cluster, agglomerative_clustering_data)
            cluster = Cluster(cluster_node_id)
            child_1.parent = cluster
            child_2.parent = cluster 
            cluster.level = self.__calculate_level([child_1, child_2])
            cluster.add_children(child_1)
            cluster.add_children(child_2)
            hierarchical_cluster_node_id_to_cluster[cluster_node_id] = cluster
            if hierarchical_cluster_node_id_to_cluster[cluster_node_id].level not in self.level_to_clusters:
                self.level_to_clusters[hierarchical_cluster_node_id_to_cluster[cluster_node_id].level] = []
            self.level_to_clusters[hierarchical_cluster_node_id_to_cluster[cluster_node_id].level].append(hierarchical_cluster_node_id_to_cluster[cluster_node_id])
            return cluster
        
    def __get_level(self, cluster):
        return cluster.level
    
    def __calculate_level(self, children):
        return max(list(map(self.__get_level, children))) + 1
    
    def __get_subreddit_label(self, subreddit):
        return subreddit.subreddit_label

    def get_data(self, include_subreddits):
        return self.__get_data_helper(self.tree, include_subreddits)
        
    def __get_data_helper(self, tree, include_subreddits):
        taxonomy_label = ""
        if type(tree.taxonomy_label) == str:
            taxonomy_label = tree.taxonomy_label
        if type(tree) is Cluster:
            children = []
            for child in tree.children:
                if type(child) == Cluster or include_subreddits:
                    children.append(self.__get_data_helper(child, include_subreddits))
            top_subreddits = ",".join(list(map(self.__get_subreddit_label, tree.get_top_subreddits("comment_count")[0:10])))
            parent_node_id = ""
            if tree.parent != None:
                parent_node_id = tree.parent.node_id
            return {"node_id": str(tree.node_id), "comment_count": str(tree.comment_count), 
                "parent": str(parent_node_id), "level": str(tree.level), "taxonomy_label": taxonomy_label, 
                "tsne_x": str(tree.tsne_x), "tsne_y": str(tree.tsne_y), "children": children, "subreddit_count": str(tree.subreddit_count), 
                "top_subreddits_by_comment": top_subreddits, "subreddit_count": str(tree.subreddit_count), 
                "color": str(self.taxonomy_label_to_color[taxonomy_label])}

        return {"node_id": str(tree.node_id) + "_" + str(tree.index), "cluster_label": str(tree.node_id), 
                "comment_count": str(tree.comment_count), 
                "parent": str(tree.parent.node_id), "level": str(tree.level), "taxonomy_label": taxonomy_label, 
                "tsne_x": str(tree.tsne_x), "tsne_y": str(tree.tsne_y), "subreddit": tree.subreddit_label, "subreddit_count": "1", 
                "color": str(self.taxonomy_label_to_color[taxonomy_label])}

"""
subreddit_count and cluster_subreddit count --> comment_count
no all_subreddit_labels
top_subreddit_labels --> top_subreddits_by_comment
addition of parent

"""

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

class MapDataSummary:
    def __init__(self, cluster_node_ids, parent_cluster_node_ids, labels, arr_of_subreddits):
        self.__cluster_node_ids = cluster_node_ids
        self.__parent_cluster_node_ids = parent_cluster_node_ids
        self.__labels = labels
        self.__arr_of_subreddits = arr_of_subreddits
        self.__num_of_kmeans_clusters = len(cluster_node_ids) 
        self.__max_hierarchical_cluster_node_id = len(cluster_node_ids) - 1
        self.__num_of_kmeans_clusters = (len(cluster_node_ids) + 1) / 2
        self.data = self.__format_data(self.__cluster_node_ids, self.__labels, self.__arr_of_subreddits, self.__parent_cluster_node_ids)
        self.label_to_subreddits = self.__get_label_to_subreddits()
        self.subreddit_to_label = self.__get_subreddit_to_label()
        self.cluster_node_id_to_subreddits = self.__get_cluster_node_id_to_subreddits()
        self.cluster_node_id_to_label = self.__get_cluster_node_id_to_label()
        self.label_to_cluster_node_id = self.__get_label_to_cluster_node_id()
        self.parent_to_children = self.__get_parent_to_children()
        self.cluster_node_id_to_level = self.__get_cluster_node_id_to_level()
        self.level_to_cluster_node_ids = self.__get_level_to_cluster_node_ids()
        self.child_to_parent = self.__get_child_to_parent()
               

    
    def __format_data(self, cluster_node_ids, labels, arr_of_subreddits, parent_cluster_node_ids):
        data = []
        for i in range(len(cluster_node_ids)):
            data.append({"cluster_node_id": cluster_node_ids[i], 
                         "label": labels[i], 
                         "subreddits": arr_of_subreddits[i], 
                         "parent_cluster_node_id": parent_cluster_node_ids[i]})

        return data

    def __get_label_to_subreddits(self):
        label_to_subreddits = {}
        for row in self.data:
            label = row["label"]
            subreddits = set(row["subreddits"].split(","))

            if label not in label_to_subreddits:
                label_to_subreddits[label] = set()
            label_to_subreddits[label].union(subreddits)
        return label_to_subreddits
            

    def __get_subreddit_to_label(self):
        subreddit_to_label = {}
        for row in self.data:
            label = row["label"]
            subreddits = row["subreddits"].split(",")
            for subreddit in subreddits:
                subreddit_to_label[subreddit] = label
        return subreddit_to_label

    def __get_cluster_node_id_to_subreddits(self):
        cluster_node_id_to_subreddits = {}
        for row in self.data:
            cluster_node_id = row["cluster_node_id"]
            label = row["subreddits"].split(",") 
            cluster_node_id_to_subreddits[cluster_node_id] = label
        return cluster_node_id_to_subreddits

    def __get_cluster_node_id_to_label(self):
        cluster_node_id_to_label = {}
        for row in self.data:
            cluster_node_id = row["cluster_node_id"]
            subreddits = row["label"]
            cluster_node_id_to_label[cluster_node_id] = subreddits
        return cluster_node_id_to_label
    
    def __get_label_to_cluster_node_id(self):
        label_to_cluster_node_id = {}
        for row in self.data:
            cluster_node_id = row["cluster_node_id"]
            label = row["label"]
            label_to_cluster_node_id[label] = cluster_node_id
        return label_to_cluster_node_id
    
    def __get_cluster_node_id_to_level(self):
        cluster_node_id_to_level = {}
        sorted_cluster_node_ids = [] 
        for cluster_node_id in self.__cluster_node_ids:
            sorted_cluster_node_ids.append(int(cluster_node_id))

        sorted_cluster_node_ids.sort()
        sorted_parent_cluster_node_ids = [self.__parent_cluster_node_ids[i] for i in range(len(self.__parent_cluster_node_ids))] 
        for i in range(len(self.__parent_cluster_node_ids)):
            parent_cluster_node_id = int(self.__parent_cluster_node_ids[i])
            cluster_node_id = int(self.__cluster_node_ids[i])
            index = sorted_cluster_node_ids.index(cluster_node_id)
            sorted_parent_cluster_node_ids[index] = parent_cluster_node_id
        for i in range(len(sorted_cluster_node_ids)):
            cluster_node_id = sorted_cluster_node_ids[i]
            parent_cluster_node_id = sorted_parent_cluster_node_ids[i]
            if cluster_node_id < self.__num_of_kmeans_clusters:
                cluster_node_id_to_level[cluster_node_id] = 0
            else:
                children = self.parent_to_children[cluster_node_id]
                child_1_level = children[0]
                child_2_level = children[1]
                if cluster_node_id_to_level[child_1_level] == cluster_node_id_to_level[child_2_level]:
                    cluster_node_id_to_level[cluster_node_id] = cluster_node_id_to_level[child_1_level] + 1
                else:
                    cluster_node_id_to_level[cluster_node_id] = max(cluster_node_id_to_level[child_1_level], cluster_node_id_to_level[child_2_level]) + 1
        
        return cluster_node_id_to_level


    def __get_level_to_cluster_node_ids(self):
        level_to_cluster_node_ids = {}
        for cluster_node_id, level in self.cluster_node_id_to_level.items():
            if level not in level_to_cluster_node_ids:
                level_to_cluster_node_ids[level] = set()
            level_to_cluster_node_ids[level].add(cluster_node_id)
        return level_to_cluster_node_ids

    def __get_parent_to_children(self):
        parent_to_children = {}
        for row in self.data:
            if int(row["parent_cluster_node_id"]) not in parent_to_children:
                parent_to_children[int(row["parent_cluster_node_id"])] = []
            parent_to_children[int(row["parent_cluster_node_id"])].append(int(row["cluster_node_id"]))
        return parent_to_children
    
    def __get_child_to_parent(self):
        child_to_parent = {}
        for row in self.data:
            child_to_parent[row["cluster_node_id"]] = row["parent_cluster_node_id"]
    
class NewGenerateMapping:
    def __init__(self, orig_tree_obj, tree_obj, human_taxonomy_df):
        self.__new_cluster = orig_tree_obj.kmeans_cluster_node_id_to_cluster[-1]
        self.intersect = set(map(self.__get_subreddit_label, orig_tree_obj.subreddits)).intersection(set(map(self.__get_subreddit_label, tree_obj.subreddits)))
        self.level_1_to_2 = {}
        self.orig_parent_to_cluster_node_ids = {}
        self.__num_of_kmeans_clusters = len(orig_tree_obj.kmeans_cluster_node_id_to_cluster) - 1
        self.__max_hierarchical_cluster_node_id = (self.__num_of_kmeans_clusters * 2) - 2
        self.__get_level_1_to_2(human_taxonomy_df)
        self.orig_child_to_parents = self.__get_child_to_parents(orig_tree_obj)
        self.parent_cluster_node_id_to_cluster_node_ids = {}
        self.cluster_node_id_to_label = {}
        self.cluster_node_id_to_orig_taxonomy_label = self.__get_matches(tree_obj, orig_tree_obj)
        
        

    def __get_child_to_parents(self, tree_obj):
        child_to_parents = {}
        self.__get_child_to_parents_helper(tree_obj.tree, child_to_parents)
        # add Porn
        for kmeans_cluster_node_id, cluster in tree_obj.kmeans_cluster_node_id_to_cluster.items():
            if cluster.taxonomy_label == "Porn" and cluster.node_id != 138:
                child_to_parents[cluster] = [tree_obj.hierarchical_cluster_node_id_to_cluster[138]]
            if kmeans_cluster_node_id == -1:
                child_to_parents[cluster] = [tree_obj.hierarchical_cluster_node_id_to_cluster[198]]
        return child_to_parents
        

    def __get_child_to_parents_helper(self, tree, child_to_parents):
        for child in tree.children:
            if type(child) == Cluster:
                if child not in child_to_parents:
                    child_to_parents[child] = []
                child_to_parents[child].append(tree)
                self.__get_child_to_parents_helper(child, child_to_parents)
            
    def __get_level_1_to_2(self, human_taxonomy_df):
        for index, row in human_taxonomy_df.iterrows():
            if row["cluster_node_id"] < self.__num_of_kmeans_clusters:
                if row["cluster_node_id"] not in self.level_1_to_2:
                    self.level_1_to_2[row["cluster_node_id"]] = []
                self.level_1_to_2[row["cluster_node_id"]].append(row["parent_cluster_node_id"])
                self.orig_parent_to_cluster_node_ids[row["parent_cluster_node_id"]] = []


    def __get_matches(self, tree_obj, orig_tree_obj):
        matches_curr_to_orig = {}
        matches_orig_to_curr = {}
        orig_cluster_node_id_to_cluster_node_id = {}
        orig_level_1_cluster_node_id_to_cluster_node_ids = {}
        curr_to_orig = {}
        orig_to_cur = {}
        set_of_labels = set()
        for kmeans_cluster_node_id, cluster in tree_obj.kmeans_cluster_node_id_to_cluster.items():
            max_kmeans_cluster_node_id = None
            max_sim = 0
            curr_to_orig[cluster] = {}
            threshold = 0.1
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
                reduce += 0.025
                threshold -= reduce
            if len(curr_to_orig[cluster]) == 0:
                curr_to_orig[cluster][self.__new_cluster] = 0
                if self.__new_cluster not in orig_to_cur:
                    orig_to_cur[self.__new_cluster] = {}
                orig_to_cur[self.__new_cluster][cluster] = 0

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
        count = 0
        while (len(matches_curr_to_orig) != self.__num_of_kmeans_clusters or  len(matches_orig_to_curr) != self.__num_of_kmeans_clusters) and count != 3:
            if count == 0 or count == 1:
                for cluster, orig_clusters in curr_to_orig.items():
                    max_orig_taxonomy_label = max(orig_clusters, key=orig_clusters.get)
                    if count == 0 and ((cluster not in matches_curr_to_orig and max_orig_taxonomy_label not in matches_orig_to_curr) or max_orig_taxonomy_label.node_id == "Porn"):
                        max_orig_taxonomy_label = max(orig_clusters, key=orig_clusters.get)
                        if max(orig_to_cur[max_orig_taxonomy_label], key=orig_to_cur[max_orig_taxonomy_label].get) == cluster or max_orig_taxonomy_label.node_id == "Porn":
                            if cluster not in matches_curr_to_orig:
                                matches_curr_to_orig[cluster] = []
                            matches_curr_to_orig[cluster].append(max_orig_taxonomy_label)
                            if max_orig_taxonomy_label not in matches_orig_to_curr:
                                matches_orig_to_curr[max_orig_taxonomy_label] = []
                            matches_orig_to_curr[max_orig_taxonomy_label].append(cluster)
                    if count == 1 and cluster not in matches_curr_to_orig:
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
            if count == 2:
                for orig_cluster, clusters in orig_to_cur.items():
                    if orig_cluster not in matches_orig_to_curr:
                        for cluster, sim in clusters.items():
                            if sim > 0.1:
                                matches_curr_to_orig[cluster].append(orig_cluster)
                                if orig_cluster not in matches_orig_to_curr:
                                    matches_orig_to_curr[orig_cluster] = []
                                matches_orig_to_curr[orig_cluster].append(cluster)
            count += 1

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
                    if child.node_id != -1:
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
                        
                    
        multi_label_cluster_node_ids = []
        for cluster_node_id, labels in self.cluster_node_id_to_label.items():
            if cluster_node_id != self.__max_hierarchical_cluster_node_id and len(labels.split("|")) > 1 and len(set(cluster_node_id_to_parent_cluster_node_ids[cluster_node_id])) > 1:
                multi_label_cluster_node_ids.append(cluster_node_id)
        multi_label_cluster_node_id_to_location = {}
        for multi_label_cluster_node_id in multi_label_cluster_node_ids:
            parents = set(cluster_node_id_to_parent_cluster_node_ids[multi_label_cluster_node_id])
            common_parent = self.__find_common_parent(cluster_node_id_to_parent_cluster_node_ids, parents)
            if common_parent != self.__max_hierarchical_cluster_node_id:
                multi_label_cluster_node_id_to_location[multi_label_cluster_node_id] = common_parent
                if multi_label_cluster_node_id not in self.parent_cluster_node_id_to_cluster_node_ids[common_parent]:
                    self.parent_cluster_node_id_to_cluster_node_ids[common_parent].append(multi_label_cluster_node_id)
        
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

class GenerateMapping:
    def __init__(self, tree_obj, orig_tree_obj):
        self.cluster_node_id_to_orig_taxonomy_label = self.__get_matches(tree_obj, orig_tree_obj)
        


    def __get_matches(self, tree_obj, orig_tree_obj):
        cluster_node_id_to_orig_taxonomy_label = {} 
        for level, clusters in tree_obj.level_to_clusters.items():
            if level in orig_tree_obj.level_to_clusters:
                for cluster in clusters:
                    max_sim = 0
                    max_sim_original_cluster = None
                    for orig_cluster in orig_tree_obj.level_to_clusters[level]:
                        sim = self.__get_similarity(orig_cluster, cluster)
                        if sim > max_sim:
                            max_sim = sim
                            max_sim_original_cluster = orig_cluster     
                    cluster_node_id_to_orig_taxonomy_label[cluster.node_id] = max_sim_original_cluster.taxonomy_label
            else:
                for cluster in clusters:
                    cluster_node_id_to_orig_taxonomy_label[cluster.node_id] = max_sim_original_cluster.taxonomy_label
        return cluster_node_id_to_orig_taxonomy_label
      

    def __get_subreddit_label(self, subreddit):
        return subreddit.subreddit_label


    def __get_similarity(self, orig_cluster, cluster):
        orig_cluster_subreddit_labels = set(map(self.__get_subreddit_label, orig_cluster.subreddits))
        cluster_subreddit_labels = set(map(self.__get_subreddit_label, cluster.subreddits))
        return len(orig_cluster_subreddit_labels
                   .intersection(cluster_subreddit_labels)) / len(orig_cluster_subreddit_labels
                                                                      .union(cluster_subreddit_labels))

def get_subreddit_label(subreddit):
    return subreddit.subreddit_label

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-m', '--month_name')
    args = parser.parse_args()


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
    orig_tree_obj_rules_mapping = NewTree(orig_clusters_df, orig_tsne_df, orig_subreddit_comment_counts_df, orig_human_taxonomy_rules_mapping_df, orig_kmeans_cluster_centers, orig_subreddit_centers, "2021-04")
    orig_tree_obj_rules_mapping.get_tree(orig_parent_cluster_node_id_to_cluster_node_id_rules_mapping)
    orig_tree_obj_rules_mapping.add_labels(orig_cluster_node_id_to_label_rules_mapping)
    orig_tree_obj_rules_mapping.reduce_tree()
    orig_tree_obj_rules_mapping.get_sorted_distances()

    for current_month in os.listdir('../data/community2vec/'):
        print(current_month)
        #if current_month in ["RC_2021-05", "RC_2023-02", "RC_2023-01"]:

        #if current_month not in ["RC_2021-04","RC_2021-05","RC_2021-06","RC_2021-07","RC_2021-08","RC_2021-09","RC_2021-10","RC_2021-11","RC_2021-12","RC_2022-01","RC_2022-02","RC_2022-03","RC_2022-12", "RC_2023-02", "RC_2023-01"]:
        kmeans_model_path = '../data/kmeans_data/' + current_month + '_kmeans/sklearn_cluster_model.joblib'
        keyed_vectors_path = '../data/community2vec/' + current_month + '/best_model/keyedVectors'
        clusters_path = '../data/kmeans_data/' + current_month + '_kmeans/clusters.csv'
        tsne_path = '../data/community2vec/' + current_month + '/best_model/tsne.csv'
        subreddit_comment_counts_path = '../data/community2vec/' + current_month + '/subreddit_counts.csv'
        kmeans_cluster_centers = joblib.load(kmeans_model_path).cluster_centers_
        subreddit_centers = gm.KeyedVectors.load(keyed_vectors_path).get_normed_vectors()
        clusters_df = pd.read_csv(clusters_path)
        tsne_df = pd.read_csv(tsne_path)
        subreddit_comment_counts_df = pd.read_csv(subreddit_comment_counts_path)
        tree_obj_rules_mapping = NewTree(clusters_df, tsne_df, subreddit_comment_counts_df, orig_human_taxonomy_rules_mapping_df, kmeans_cluster_centers, subreddit_centers, current_month[3:])

        mapping = NewGenerateMapping(orig_tree_obj_rules_mapping, tree_obj_rules_mapping, orig_human_taxonomy_rules_mapping_df)
        tree_obj_rules_mapping.get_tree(mapping.parent_cluster_node_id_to_cluster_node_ids)
        tree_obj_rules_mapping.add_labels(mapping.cluster_node_id_to_label)
        tree_obj_rules_mapping.reduce_tree()
        tree_obj_rules_mapping.get_sorted_distances()
        tree_obj_rules_mapping.get_data(False)
        
        
        with open("../data/mapping_data/results/" + current_month + ".json", "w") as outfile:
            json.dump(tree_obj_rules_mapping.get_data(True), outfile)
        with open("../data/mapping_data/results/" + current_month + "_No_Children.json", "w") as outfile:
            json.dump(tree_obj_rules_mapping.get_data(False), outfile)
    """
    with open("../data/mapping_data/RC_2022-02_KMeans_Agglom_100_Clusters_Rules_Mapping_No_Children.json", "w") as outfile:
        json.dump(tree_obj_rules_mapping.get_data(False), outfile)
    with open("../data/mapping_data/RC_2021-04_KMeans_Agglom_100_Clusters_Rules_Mapping.json", "w") as outfile:
        json.dump(orig_tree_obj_rules_mapping.get_data(True), outfile)
    with open("../data/mapping_data/RC_2021-04_KMeans_Agglom_100_Clusters_Rules_Mapping_No_Children.json", "w") as outfile:
        json.dump(orig_tree_obj_rules_mapping.get_data(False), outfile)

    
    parent_cluster_node_id_to_cluster_node_id = {}
    orig_cluster_node_id_to_label = {}
    orig_agglomerative_clustering_data = []
    for index, row in orig_parent_df.iterrows():
        if row["parent_cluster_node_id"] not in parent_cluster_node_id_to_cluster_node_id:
            parent_cluster_node_id_to_cluster_node_id[row["parent_cluster_node_id"]] = []
        parent_cluster_node_id_to_cluster_node_id[row["parent_cluster_node_id"]].append(row["cluster_node_id"])
    for index, row in orig_human_taxonomy_df.iterrows():
        orig_cluster_node_id_to_label[row["cluster_node_id"]] = row["label"]
    for i in range(len(parent_cluster_node_id_to_cluster_node_id) - 1):
        orig_agglomerative_clustering_data.append(parent_cluster_node_id_to_cluster_node_id[len(parent_cluster_node_id_to_cluster_node_id) + i])
    
    
    orig_tree_obj = Tree(orig_clusters_df, orig_tsne_df, orig_subreddit_comment_counts_df, orig_agglomerative_clustering_data, orig_human_taxonomy_df)
    orig_tree_obj.add_labels(orig_cluster_node_id_to_label)

    kmeans_to_agglom_object = KMeansToAgglomerative(kmeans_model_path)
    agglomerative_clustering_data = kmeans_to_agglom_object.agglomerative_clustering_data
    tree_obj = Tree(clusters_df, tsne_df, subreddit_comment_counts_df, agglomerative_clustering_data, orig_human_taxonomy_df)

    matching = GenerateMapping(tree_obj, orig_tree_obj)
    tree_obj.add_labels(matching.cluster_node_id_to_orig_taxonomy_label)
    tree_obj.reduce_tree(tree_obj.tree)
    orig_tree_obj.reduce_tree(orig_tree_obj.tree)
    orig_tree_obj.group_locals(orig_tree_obj.tree)
    tree_obj.group_locals(tree_obj.tree)
    tree_obj.tree.taxonomy_label = ""
    orig_tree_obj.tree.taxonomy_label = ""

    
    with open("../data/mapping_data/RC_2021-04_KMeans_Agglom_100_Clusters_Updated_Mapping_No_Children.json", "w") as outfile:
        json.dump(orig_tree_obj.get_data(False), outfile)

    with open("../data/mapping_data/RC_2021-05_KMeans_Agglom_100_Clusters_Updated_Mapping_No_Children.json", "w") as outfile:
        json.dump(tree_obj.get_data(False), outfile)

    with open("../data/mapping_data/RC_2021-04_KMeans_Agglom_100_Clusters_Updated_Mapping.json", "w") as outfile:
        json.dump(orig_tree_obj.get_data(True), outfile)

    with open("../data/mapping_data/RC_2021-05_KMeans_Agglom_100_Clusters_Updated_Mapping.json", "w") as outfile:
        json.dump(tree_obj.get_data(True), outfile)
    
    """

if __name__ == "__main__":
    main()