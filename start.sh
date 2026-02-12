#!/bin/bash
# Quantum Field Agent - Railway Start Script
cd backend
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
