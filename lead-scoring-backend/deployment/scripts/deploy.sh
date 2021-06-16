#!/usr/bin/env sh

set -xe
[[ "$TRACE" ]] && set -x
  export CI_APPLICATION_TAG=$CI_COMMIT_SHA

echo "Switch working directory..."

cd $HELM_CHART_PATH

echo "Start to upgrade..."

echo $$CI_APPLICATION_TAG

helm secrets upgrade --install $PROJECT ./  \
      --namespace=$KUBE_NAMESPACE --create-namespace \
      --set image.tag="$CI_APPLICATION_TAG" \
      -f deploy/$environment/secrets.yaml \
      -f deploy/$environment/values.yaml