pip install uv

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

GIT_PREFIX="https://github.com"
if [ -n "$GH_TOKEN" ]; then
  GIT_PREFIX="https://${GH_TOKEN}@github.com"
fi

git clone ${GIT_PREFIX}/kaylode/ehrpfn.git ehrpfn-inference
git clone ${GIT_PREFIX}/kaylode/ehr-meds.git
git clone -b v2.0-update ${GIT_PREFIX}/kaylode/theseus.git
