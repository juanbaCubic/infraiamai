
usage()
{
    echo "usage: script.sh [[[-n --name ] [-p --path] [-e --env_params]] | [-h --help]]"
    echo "* -n:   name of the cloudformation stack which will be created"
    echo "* -p:   path to the cloudformation stack file description"
    echo "* -e:   direct environment variables"
    echo ""
    echo "To use this script you must be logged in aws cli and be in AmazonIAMPowerUser role or AmazonIAMFullAccess https://docs.aws.amazon.com/es_es/IAM/latest/UserGuide/access_policies_managed-vs-inline.html"
    echo ""
    echo "example:   ./exec-cloudformation.sh -n prueba-rol -p ./role-creation.yaml -e 'ParameterKey=Prefix,ParameterValue=prol'"
    exit 1
}

while [ "$1" != "" ]; do
    case $1 in
        -n | --name )       shift
                                name=$1
                                ;;
        -p | --path )       shift
                                path=$1
                                ;;
        -r | --region )     shift
                                REGION=$1
                                ;;
        -e | --env_params )     shift
                                env_params=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


if [[ -n "$REGION" ]]; then
  export AWS_REGION="$REGION"
elif [[ -n "$AWS_REGION" ]]; then
  export AWS_REGION="$AWS_REGION"
else
  export AWS_REGION="$(aws configure get region)"
fi


existstack=$(aws cloudformation describe-stacks --region "${AWS_REGION}" | jq  -r  ".Stacks[] | select (.StackName==\"${name}\")")
if test -n "$existstack"
then
    StackID=$(aws cloudformation update-stack \
        --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND\
        --stack-name ${name} \
        --template-body file://./${path} \
        --region "${AWS_REGION}" \
        --parameters $env_params)
else
    StackID=$(aws cloudformation create-stack \
        --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND\
        --stack-name ${name} \
        --template-body file://./${path} \
        --region "${AWS_REGION}" \
        --parameters $env_params)
fi

echo $StackID

