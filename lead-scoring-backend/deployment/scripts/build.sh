#!/usr/bin/env sh

set -e

  [[ "$TRACE" ]] && set -x
    export CI_APPLICATION_REPOSITORY=$CI_REGISTRY_IMAGE
    export CI_APPLICATION_TAG=$CI_COMMIT_SHA
    export CI_CONTAINER_NAME=ci_job_build_${CI_JOB_ID}

  legacy="${2-"main"}"

  if [[ $legacy == "legacy" ]]; then
    CI_APPLICATION_REPOSITORY=$CI_REGISTRY_IMAGE/legacy
  fi

  echo "Logging to GitLab Container Registry with CI credentials..."
  docker login -u "$CI_REGISTRY_USER" -p "$CI_REGISTRY_PASSWORD" "$CI_REGISTRY"
  echo ""

  docker build -t "$CI_APPLICATION_REPOSITORY:$CI_APPLICATION_TAG" -t "$CI_APPLICATION_REPOSITORY:$ENVIRONMENT-latest" .

  echo "Pushing to GitLab Container Registry..."
  docker push "$CI_APPLICATION_REPOSITORY:$CI_APPLICATION_TAG"
  docker push "$CI_APPLICATION_REPOSITORY:$ENVIRONMENT-latest"
  echo ""
