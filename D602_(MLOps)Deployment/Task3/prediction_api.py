#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException
import json
import numpy as np
import pickle
import datetime

# Import the airport encodings file
f = open('airport_encodings.json')
 
# returns JSON object as a dictionary
airports = json.load(f)

def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.  

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport
        
    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1 
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        temp = temp.T
        return temp
    else:
        return None

# TODO:  write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded 
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE

with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_delay(arrival_airport, departure_time, arrival_time):
    encoded_airport = create_airport_encoding(arrival_airport, airports)
    if encoded_airport is None:
        raise HTTPException(status_code=404, detail="Arrival airport not found")

    try:
        # Parse the times using the correct format.
        dep_dt = datetime.datetime.strptime(departure_time, "%Y-%m-%dT%H:%M:%S")
        arr_dt = datetime.datetime.strptime(arrival_time, "%Y-%m-%dT%H:%M:%S")

        # Calculate seconds since midnight.
        dep_time_seconds = dep_dt.hour * 3600 + dep_dt.minute * 60 + dep_dt.second
        arr_time_seconds = arr_dt.hour * 3600 + arr_dt.minute * 60 + arr_dt.second
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Please use 'YYYY-MM-DDTHH:MM:SS'.")

    input_data = np.concatenate(([1], encoded_airport, [dep_time_seconds], [arr_time_seconds]))

    delay = model.predict(input_data.reshape(1, -1))
    return delay[0]

# TODO:  write the API endpoints.  
# YOUR CODE GOES HERE

#Initializing FastAPI app
app = FastAPI()

#Root endpoint to check if the API is functional
@app.get("/")
async def root():
    return {"message": "API is functional!"}

#Prediction endpoint to get the average departure delay
@app.get("/predict/delays")
async def predict_delays(arrival_airport: str, departure_time: str, arrival_time: str):
    try:
        delay = predict_delay(arrival_airport, departure_time, arrival_time)
        return {"average_departure_delay": delay}
    except HTTPException as e:
        raise e