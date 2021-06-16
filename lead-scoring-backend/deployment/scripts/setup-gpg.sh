#!/usr/bin/env sh

# setup environment
export GPG_TTY=$(tty)

# ensure that GPG pass is available
if [[ -z ${HELM_SECRETS_PASS+x} ]]; then
    echo "HELM_SECRETS_PASS is not set";
    exit 1;
fi

# Import GPG key to container
echo "${HELM_SECRETS_PASS}" | \
gpg2 --pinentry-mode loopback \
     --passphrase-fd 0 \
     --import $GPG_PRIV_KEY_PATH

# hack to unlock GPG key
touch /tmp/dummy && echo "${HELM_SECRETS_PASS}" | \
gpg2 --pinentry-mode loopback \
     --passphrase-fd 0 \
     -s /tmp/dummy
