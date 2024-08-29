FROM python:3.11

COPY ./pricing/
COPY ./requirements.txt / 

WORKDIR /pricing

RUN pip install -r requirements.txt

EXPOSE 1234

ENTRYPOINT ["PricingProject"]
CMD ["serve", "--dev-addr=0.0.0.0:1234"]