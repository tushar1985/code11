FROM python:3.8
                                                                                                                                                                                                                                 
WORKDIR /training/Training_data_generation                                                                                                                                                                                                               

COPY ./requirements.txt /requirements.txt                                                                                                                                                                                                       

COPY . .                                                                                                                                                                                                                                        

RUN apt-get install libpq-dev                                                                                                                                                                                                                   

RUN pip3 install -U pip wheel cmake                                                                                                                                                                                                             

RUN pip3 install psycopg2-binary                                                                                                                                                                                                                

RUN pip3 install --no-cache -r requirements.txt                                                                                                                                                                                                 

RUN apt-get update && apt-get install -y iputils-ping                                                                                                                                                                                           

EXPOSE 2000                                                                                                                                                                                                                                     

CMD [ "python" , "my_app.py" ]  