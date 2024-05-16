#!/bin/bash

set -eu

# install quelware
uribase=https://github.com/quel-inc/quelware/releases/download
version=0.8.8
firmware=simplemulti_standard
archivename=quelware_prebuilt.tgz

rootdir=$(dirname $(dirname $(realpath $0)))
mkdir -p "${rootdir}/lib/quelware"
cd "${rootdir}/lib/quelware"

echo "INFO: downloading ${uribase}/${version}/${archivename} ..."
wget -q "${uribase}/${version}/${archivename}" -O ${archivename} || (echo "ERROR: no prebuilt archive is available for ${version}" && exit 1)

echo "INFO: extracting ${archivename}..."
tar -xzf ${archivename}

echo "INFO: installing quelware..."
python -m pip install -r "requirements_${firmware}.txt"

# install qubecalib
echo "INFO: installing qubecalib..."
cd $rootdir
pip install .
echo "INFO: installation completed"