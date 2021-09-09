make clean && make all && make package && \
docker push andazhou/determined-master:0.16.3.dev0 && \
docker push andazhou/determined-agent:0.16.3.dev0 && \
det deploy aws down --cluster-id anda && \
det deploy --image-repo-prefix andazhou aws up --cluster-id anda --keypair anda-determined

