# RedditMap Data
Publishing Scripts for Data Backing the RedditMap Project

## Background

RedditMap is a project of computer, data, and social scientists to explore and visualize Reddit as a space. The site running the visualizations is available at https://redditmap.social. But the data that backs that site is stored here.

## Main Tasks

### Dependencies
Install dependencies with `npm install`.

### Publishing Changes to Data

After processing and preparing data, write it to the directory `/data/redditmap`. We treat that directory as the authority. The published data will be available just as it appears there. However, we currently have an exception for the remote `/images` subdirectory. Please do not create a local `/data/redditmap/images` subdirectory because it will be ignored.

After you commit changes to our mainline branch, `main`, you'll want to cut a release and publish changes for the world to see. We've setup a GitHub action to make that easy for you. The publish task will sync storage in our infrastructure so that it looks exactly like what's in that directory. So anything you put into `/data/redditmap` will be mirrored remotely.

Merge the branch `main` into `publish-redditmap`. The publish action will take care of the rest for you, syncing remote files, and dealing with HTTP caching. It will take several minutes for the sync and HTTP cache invalidation tasks to complete. 


## Other Tasks

There are Gulp tasks that allow you to perform other actions, but most people won't need to use them if they're working directly on the data layer.


### Deploying Infrastructure

You only need this if you're setting up an endpoint for the first time, or if the infrastructure configuration needs to be updated. Note that you'll need sufficient permissons in iDPI's AWS account.

```bash
AWS_PROFILE=idpi npx gulp deploy --environment=production-data
```


### Manually Syncing Data

You can run the publish action locally, if you have permissions with iDPI's AWS account. Use the following command:

```bash
AWS_PROFILE=idpi npx gulp publish --environment=production-data
```