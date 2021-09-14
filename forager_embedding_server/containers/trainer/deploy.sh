project=`gcloud config get-value project 2> /dev/null`
folder=.
name=forager-index-${PWD##*/}
root_path=../../..

# Copy shared resources in
cp -r $root_path/interactive_index/interactive_index $folder

# Submit build from within directory
gcloud config set builds/use_kaniko True
(cd $folder; gcloud builds submit --tag gcr.io/$project/$name)

# Remove shared resources
rm -rf $folder/interactive_index
