import * as Task from "./tasks/index.mjs";


const deploy = async function () {
  const config = await Task.Environment.check();
  await Task.Bucket.deploy( config );
  await Task.Edge.deploy( config );
};

const teardown = async function () {
  const config = await Task.Environment.check();
  await Task.Edge.teardown( config );
  await Task.Bucket.teardown( config );
};

const publish = async function () {
  const config = await Task.Environment.check();
  const files = await Task.Directory.read( config[ "bucket-sync" ].directory );
  await Task.Bucket.check( config );
  await Task.Bucket.sync( config, files );

  // Only cache invalidate for environments that use caching.
  if ( config.edge.ttl.default > 0 ) {
    await Task.Edge.invalidate( config );
  }
};


export {
  deploy,
  teardown,
  publish
}