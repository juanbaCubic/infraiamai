#!/usr/bin/env bash

# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
set -uo pipefail

usage() { echo "Usage: $0 [-p profile] [-r region] [-D]" 1>&2; exit 1; }

UUID="$RANDOM"
REGION=''
PROFILE=''
DELETE_STACKS_ON_ERROR=1

while getopts ":p:r:D" o; do
    case "${o}" in
        D)
            DELETE_STACKS_ON_ERROR=0
            ;;
        r)
            REGION=${OPTARG}
            ;;
        p)
            PROFILE=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))


# the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# the temp directory used, within $DIR
# omit the -p parameter to create a temporal directory in the default location
WORK_DIR=`mktemp -d`


if [[ -n "$PROFILE" ]]; then
  export AWS_PROFILE="$PROFILE"
fi


if [[ -n "$REGION" ]]; then
  export AWS_REGION="$REGION"
else
  export AWS_REGION="$(aws configure get region)"
fi

NAME_STACK_CONFIG="$1-$UUID"
NAME_STACK_LAMBDA="$2-$UUID"
PREFIX="$3-$UUID"
#URL_LAMBDA_CONTAINER=""$WORK_DIR"/lambda_function/"
URL_LAMBDA_CONTAINER="lambda_function/"
STACK_IDS=()

echo $DIR

cp -r $URL_LAMBDA_CONTAINER $WORK_DIR

# check if tmp dir was created
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
fi

# deletes the temp directory
function cleanup {
  EXIT_CODE=$?
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
  if [[ "$DELETE_STACKS_ON_ERROR" -eq 1 && "$EXIT_CODE" -ne 0 ]]; then
    for stackid in ${STACK_IDS[@]}; do
      echo "Deleting stack ${stackid} ..."
      aws cloudformation delete-stack --stack-name "${stackid}" --region "${AWS_REGION}"
    done
  fi
  exit $EXIT_CODE
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT INT TERM

StackID=$(./exec-cloudformation.sh -n $NAME_STACK_CONFIG -p ./role-creation.yaml -r "${AWS_REGION}" \
-e "ParameterKey=Prefix,ParameterValue=$PREFIX \
ParameterKey=TopicRegion,ParameterValue=$AWS_REGION")

# check if command succeeded
if [[ $? -ne 0 || "$StackID" == "" ]]; then
  echo "Couldn't obtain StackID"
  exit 1
fi

StackID=$(echo $StackID | jq -r '.StackId')
# check if command succeeded
if [[ $? -ne 0 || "$StackID" == "" ]]; then
  echo "Could obtain StackID"
  exit 1
fi
echo "StackID: $StackID"
STACK_IDS+=("$StackID")


#aws cloudformation wait stack-create-complete --stack-name $StackID --region "${AWS_REGION}"
# Wait for create-stack to finish
echo  "Waiting for create-stack $NAME_STACK_CONFIG command to complete"
CREATE_STACK_STATUS=$(aws cloudformation describe-stacks --stack-name "${NAME_STACK_CONFIG}" --query 'Stacks[0].StackStatus' --output text --region "${AWS_REGION}")
while [[ "$CREATE_STACK_STATUS" == "UPDATE_IN_PROGRESS" ]] || [[ "$CREATE_STACK_STATUS" == "CREATE_IN_PROGRESS" ]]
do
    # Wait 30 seconds and then check stack status again
    sleep 30
    CREATE_STACK_STATUS=$(aws cloudformation describe-stacks --stack-name "${NAME_STACK_CONFIG}" --query 'Stacks[0].StackStatus' --output text --region "${AWS_REGION}")
    aws cloudformation describe-stack-events --stack-name "$StackID" --region "${AWS_REGION}" | jq '.StackEvents | reverse | reduce .[] as $d ({}; .[$d.LogicalResourceId] = $d.ResourceStatus) | to_entries[] | select(.value != "CREATE_COMPLETE")'
done

if [[ "$CREATE_STACK_STATUS" != "CREATE_COMPLETE" ]] ; then
  echo "Stack $StackID failed to complete: $CREATE_STACK_STATUS"
  aws cloudformation describe-stack-events --stack-name "$StackID" --region "${AWS_REGION}" | jq '.StackEvents[] | select((.ResourceStatus | contains("FAILED")) or (.ResourceStatus | contains("ROLLBACK")))'
  exit 1
fi

repoID=$(aws cloudformation describe-stacks --stack-name "$StackID" --query "Stacks[0].Outputs[?OutputKey=='ECRCubicRepoID'].OutputValue" --output text --region "${AWS_REGION}")

nameAccount=$(aws sts get-caller-identity --query Account --output text --region "${AWS_REGION}")

echo 'Stack 1 Created'

cd $WORK_DIR/lambda_function/
echo "Downloading model and dictionary to $URL_LAMBDA_CONTAINER"

wget "http://sciling.com/img/uploads.zip" -O temp1.zip
wget "http://sciling.com/img/data.zip" -O temp2.zip

#mv temp1.zip  temp2.zip $URL_LAMBDA_CONTAINER

#cd $URL_LAMBDA_CONTAINER

unzip -o temp1.zip
unzip -o temp2.zip
rm temp1.zip temp2.zip

cd -
pwd
ls $WORK_DIR/lambda_function/

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${nameAccount}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker build "$WORK_DIR/lambda_function/" --tag "$repoID"
docker push "$repoID"

StackID2=$(./exec-cloudformation.sh -n "$NAME_STACK_LAMBDA" -p ./lambda-trigger.yaml -e "ParameterKey=Prefix,ParameterValue=$PREFIX ParameterKey=ArchitechtureName,ParameterValue=$NAME_STACK_CONFIG" -r "${AWS_REGION}")

# check if command succeeded
if [[ $? -ne 0 || "$StackID2" == "" ]]; then
  echo "Couldn't obtain StackID2"
  exit 1
fi

StackID2=$(echo $StackID2 | jq -r '.StackId')
# check if command succeeded
if [[ $? -ne 0 || "$StackID2" == "" ]]; then
  echo "Could obtain StackID2"
  exit 1
fi
echo "StackID2: $StackID2"
STACK_IDS+=("$StackID2")

# Wait for create-stack to finish
echo  "Waiting for create-stack $StackID2 command to complete"
CREATE_STACK_STATUS=$(aws cloudformation describe-stacks --stack-name "${StackID2}" --query 'Stacks[0].StackStatus' --output text --region "${AWS_REGION}")
while [[ "$CREATE_STACK_STATUS" == "UPDATE_IN_PROGRESS" ]] || [[ "$CREATE_STACK_STATUS" == "CREATE_IN_PROGRESS" ]]
do
    # Wait 30 seconds and then check stack status again
    sleep 30
    CREATE_STACK_STATUS=$(aws cloudformation describe-stacks --stack-name "${StackID2}" --query 'Stacks[0].StackStatus' --output text --region "${AWS_REGION}")
    aws cloudformation describe-stack-events --stack-name "${StackID2}" --region "${AWS_REGION}" | jq '.StackEvents | reverse | reduce .[] as $d ({}; .[$d.LogicalResourceId] = $d.ResourceStatus) | to_entries[] | select(.value != "CREATE_COMPLETE")'
done

if [[ "$CREATE_STACK_STATUS" != "CREATE_COMPLETE" ]]; then
  echo "Stack $StackID2 failed to complete: $CREATE_STACK_STATUS"
  aws cloudformation describe-stack-events --stack-name "$StackID2" --region "${AWS_REGION}" | jq '.StackEvents[] | select((.ResourceStatus | contains("FAILED")) or (.ResourceStatus | contains("ROLLBACK")))'
  exit 1
fi


echo 'Stack 2 Created'
