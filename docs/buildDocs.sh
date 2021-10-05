#!/bin/bash
set -x

apt-get update

apt-get -y install python3-pip
python3 -m pip install Sphinx==4.1.2
apt-get -y install git rsync build-essential python3-stemmer python3-gitpython3-virtualenv python3-setuptools
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade rinohtype pygments

pwd
ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

# make a new temp dir which will be our GitHub Pages docroot
docroot=`mktemp -d`
 
export REPO_NAME="${GITHUB_REPOSITORY##*/}"
  
##############
# BUILD DOCS #
##############

make -C docs clean
versions="`git for-each-ref '--format=%(refname:lstrip=-1)' refs/remotes/origin/ | grep -viE '^(HEAD|gh-pages)$'`"
for current_version in ${versions}; do
  
   # make the current language available to conf.py
   export current_version
   git checkout ${current_version}
  
   echo "INFO: Building sites for ${current_version}"
  
   # skip this branch if it doesn't have our docs dir & sphinx config
   if [ ! -e 'docs/conf.py' ]; then
      echo -e "\tINFO: Couldn't find 'docs/conf.py' (skipped)"
      continue
   fi

    # HTML #
    sphinx-build -b html docs/ docs/_build/html/${current_version}

    # PDF #
    sphinx-build -b rinoh docs/ docs/_build/rinoh
    mkdir -p "${docroot}/${current_version}"
    cp "docs/_build/rinoh/target.pdf" "${docroot}/${current_version}/helloWorld-docs__${current_version}.pdf"

    # EPUB #
    sphinx-build -b epub docs/ docs/_build/epub
    mkdir -p "${docroot}/${current_version}"
    cp "docs/_build/epub/target.epub" "${docroot}/${current_version}/helloWorld-docs_${current_version}.epub"

    # copy the static assets produced by the above build into our docroot
    rsync -av "docs/_build/html/" "${docroot}/"

  
done
 
 # return to master branch
git checkout master

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
 
pushd "${docroot}"

git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages
 
touch .nojekyll
 
 # add redirect from the docroot to our default docs language/version
cat > index.html <<EOF
<!DOCTYPE html>
<html>
   <head>
      <title>helloWorld Docs</title>
      <meta http-equiv = "refresh" content="0; url='/${REPO_NAME}/en/master/'" />
   </head>
   <body>
      <p>Please wait while you're redirected to our <a href="/${REPO_NAME}/en/master/">documentation</a>.</p>
   </body>
</html>
EOF

cat > README.md <<EOF
# GitHub Pages Cache
 
Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.
 
 
If you're looking to update our documentation, check the relevant development branch's 'docs/' dir.
EOF

git add .
 
# commit all the new files
msg="Updating Docs for commit ${GITHUB_SHA} made on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"
 
# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force
 
popd # return to main repo sandbox root
 
# exit cleanly
exit 0