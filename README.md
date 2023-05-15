# Cubic Lambda Tag Invoices

**Instalar Función Lambda en AWS:**

La máquina en la que se ejecute el script debe de estar habilitada con credenciales de AWS CLI para poder realizar las operaciones (recomendable una cuenta root o con permisos necesarios: ECR, S3, Lambda, CloudFormation...)

Para instalar la función es necesario ejecutar install.sh, cuando se realice la llamada hay que pasarle tres argumentos con los nombres que se le quieran dar a las pilas de CloudFormation y un prefijo para identificar todos los elementos que se creen en ellas:

```plain
    chmod +x install.sh
    ./install.sh  ejemplo-config ejemplo-lambda ejemplo1
``` 
Los argumentos corresponden a estas variables:

```plain
    NAME_STACK_CONFIG=$1
    NAME_STACK_LAMBDA=$2
    PREFIX=$3
```
**_NAME_STACK_CONFIG_** Nombre para la pila de CloudFormation que creará el topic SNS, la cola SQS y el Rol que se necesitarán para poner en marcha la función Lambda contenida en Docker. (La imagen de Docker se construye/actualiza y se sube a ECR automáticamente desde `install.sh`)

**_NAME_STACK_LAMBDA_** Nombre para la pila de CloudFormation que creará la aquitectura para Lambda. Esta se activará con un mensaje de SNS y cuando termine los resultados se mandarán en un mensaje a una cola SQS.

**_PREFIX_** prefijo que tendrán todos los elementos creados para unificarlos.

Para enviar mensajes desde la terminal (en la interfaz web se deberá buscar el ARN del Topic SNS "TriggerTopicLambdaCubic" que creará la plantilla para poder usarlo en el siguiente comando):
```plain
    aws sns publish \
    --region eu-west-1 \
    --topic-arn "arn:aws:sns:eu-west-1:XXXXX:EJEMPLOTriggerTopicLambdaCubic" \
    --message '{"url": "https://stelnerone.s3.eu-west-1.amazonaws.com/flattened/A2021-A4.pdf"}'
```

Para recibir mensajes desde la terminal (en la interfaz web se deberá buscar la URL de la Cola SQS "ResultsLambdacucbicQueue" que creará la plantilla para poder usarlo en el siguiente comando):
```plain
    aws sqs receive-message \
    --region eu-west-1 \
    --queue-url https://sqs.eu-west-1.amazonaws.com/XXXXXXXX/EJEMPLOResultsLambdacucbicQueue
```

Estas mismas acciones se pueden hacer desde la interfaz gráfica de AWS sin hacer uso de comandos.
