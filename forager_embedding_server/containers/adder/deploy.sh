project=`gcloud config get-value project 2> /dev/null`
folder=.
name=forager-index-${PWD##*/}
root_path=../../..

# Copy shared resources in
cp -r $root_path/interactive_index/interactive_index $folder
cp -r $root_path/mihir_run/src/knn $folder

# Submit build from within subdirectory
gcloud config set builds/use_kaniko True
(cd $folder; gcloud builds submit --tag gcr.io/$project/$name)

# Remove shared resources
rm -rf $folder/interactive_index
rm -rf $folder/knn
