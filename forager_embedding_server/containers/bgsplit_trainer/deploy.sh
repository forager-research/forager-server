project=`gcloud config get-value project 2> /dev/null`
folder=.
name=forager-bgsplit-trainer
root_path=../../..

# Submit build from within directory
gcloud config set builds/use_kaniko True
(cd $folder; gcloud builds submit --tag gcr.io/$project/$name --machine-type=N1_HIGHCPU_32)
