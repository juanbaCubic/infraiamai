# Cubic Lambda Api

Actualizar imagen y lanzar docker en local:

```plain

 docker build . --tag cubic-api
 docker run --rm -p 9000:8080 -v /home/XXX/.aws:/root/.aws cubic-api
  
``` 

Es necesario pasarle dónde están alojadas las credenciales de aws para poder lanzarlo en local.

Se debe pasar un evento SNS como el del ejemplo _event.json_

```plain

cat event.json | curl -X POST -H ' content-type: application/json' -d @- "http://localhost:9000/2015-03-31/functions/function/invocations"
  
```
