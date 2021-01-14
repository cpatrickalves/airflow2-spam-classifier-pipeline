#!/bin/sh
airflow scheduler --daemon &
airflow webserver --port 8080