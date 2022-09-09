# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from cgitb import handler
import grpc
import agent_pb2
import agent_pb2_grpc
import os
import json
import sys
import array
import traceback
import numpy as np
import pandas as pd
import time
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

## Additions for Optimizer
from scipy.optimize import dual_annealing
import pandas as pd
import timeit
import uuid

# Uncomment this for grpc debug log
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'

agent_socket = "unix:///tmp/aws.greengrass.SageMakerEdgeManager.sock"

agent_channel = grpc.insecure_channel(
    agent_socket, options=(("grpc.enable_http_proxy", 0),)
)

agent_client = agent_pb2_grpc.AgentStub(agent_channel)

main_loop_on = True
num_secs = 20
num_tries = 10
model_name = ""
result_topic = ""
handle = None
COMPONENT_CLIENTID = "WindFarmOptimizer"
model_loaded = True
model_unload = False
model_url = ""
g_max_iterations = 2000     # Maximum iterations defaulted

def list_models():
    try:
        return agent_client.ListModels(agent_pb2.ListModelsRequest())
    except:
        e = sys.exc_info()[0]
        print("WindFarm_Optimizer: Exception in list_models: " + str(e))
        print(traceback.format_exc())
        return


def list_model_tensors(models):
    return {
        model.name: {
            "inputs": model.input_tensor_metadatas,
            "outputs": model.output_tensor_metadatas,
        }
        for model in list_models().models
    }


def load_model(name, url):
    try:
        load_request = agent_pb2.LoadModelRequest()
        load_request.url = url
        load_request.name = name
        return agent_client.LoadModel(load_request)
    except:
        e = sys.exc_info()[0]
        print("WindFarm_Optimizer: Exception in load_model: " + str(e))
        print(traceback.format_exc())
        return


def unload_model(name):
    try:
        unload_request = agent_pb2.UnLoadModelRequest()
        unload_request.name = name
        return agent_client.UnLoadModel(unload_request)
    except:
        e = sys.exc_info()[0]
        print("WindFarm_Optimizer: Exception in unload_model: " + str(e))
        print(traceback.format_exc())
        return


def predict(model_name, input_data):
    global model_unload
    if model_unload == True:
        try:
            # DEBUG
            # print("WindFarm_Optimizer: calling predict() on model: " + model_name + " with Input0: " + str(input_data))

            # Format the input data...
            X = (input_data["speed"]-4.0)/11.0
            Y = (input_data["direction"]-250.0)/40.0
            Z = (input_data["turbine0_yaw_angle"])/25.0
            A = (input_data["turbine1_yaw_angle"])/25.0
            np_input_data = [X, Y, Z, A]
            
            # Prepare the input tensor
            input_tensor = agent_pb2.Tensor()
            np_input_array = np.array(np_input_data).astype(np.float32)
            dimmed_np_input_array = np.expand_dims(np_input_array, axis=0)
            input_tensor.byte_data = dimmed_np_input_array.tobytes()
            input_tensor_metadata = list_model_tensors(list_models())[model_name]['inputs'][0]
            input_tensor.tensor_metadata.name = input_tensor_metadata.name
            input_tensor.tensor_metadata.data_type = input_tensor_metadata.data_type
            for shape in input_tensor_metadata.shape:
                input_tensor.tensor_metadata.shape.append(shape)
            
            # Create the predict() request
            predict_request = agent_pb2.PredictRequest()
            predict_request.name = model_name
            predict_request.tensors.append(input_tensor)

            # Call predict() and get the response
            predict_response = agent_client.Predict(predict_request)

            # DEBUG
            #print("WindFarm_Predictor: predict() raw Response: ")
            #print(predict_response)

             # Extract the ouput tensor that we'll use for our formatted response...
            for tensor in predict_response.tensors:
                # We have an output tensor
                output_array = array.array('f', tensor.byte_data)
                output_data = output_array.tolist()
                for ix in range(4):
                    output_data[ix] = output_data[ix]*5000.0

            # DEBUG
            #print("WindFarm_Optimizer: predict() numeric result - Input_Data: " + str(input_data))
            #print("WindFarm_Optimizer: predict() numeric result - Input0: " + str(dimmed_np_input_array) + " --> Output: " + str(output_data))

            # Format our response
            fmt_predict_response = {}
            fmt_predict_response['speed'] = round(input_data['speed'],2)
            fmt_predict_response['direction'] = round(input_data['direction'],2)
            fmt_predict_response['turbine0_yaw_angle'] = round(input_data['turbine0_yaw_angle'],2)
            fmt_predict_response['turbine1_yaw_angle'] = round(input_data['turbine1_yaw_angle'],2)
            for ix in range(4):
                fmt_predict_response['turbine{}_power'.format(ix)] = round(output_data[ix],2)
            
            # Return our response
            return fmt_predict_response
        except:
            e = sys.exc_info()[0]
            print("WindFarm_Optimizer: Exception in predict: " + str(e))
            print(traceback.format_exc())
            fmt_predict_response = {}
            fmt_predict_response['speed'] = round(input_data['speed'],2)
            fmt_predict_response['direction'] = round(input_data['direction'],2)
            fmt_predict_response['turbine0_yaw_angle'] = round(input_data['turbine0_yaw_angle'],2)
            fmt_predict_response['turbine1_yaw_angle'] = round(input_data['turbine1_yaw_angle'],2)
            return fmt_predict_response
    else:
        print("WindFarm_Optimizer: No models loaded")
        fmt_predict_response = {}
        fmt_predict_response['speed'] = round(input_data['speed'],2)
        fmt_predict_response['direction'] = round(input_data['direction'],2)
        fmt_predict_response['turbine0_yaw_angle'] = round(input_data['turbine0_yaw_angle'],2)
        fmt_predict_response['turbine1_yaw_angle'] = round(input_data['turbine1_yaw_angle'],2)
        return fmt_predict_response

def connect_iotcore(gg_install_root,gg_endpoint_uuid,root_cert_filename,gg_endpoint_url):
    # Custom connection info for our GG device
    ENDPOINT = gg_endpoint_url

    # if we are running GGV2 via CloudFormation, we adjust these
    if gg_endpoint_uuid == "cloudformation":
        # With Cloud Formation deployed GGv2, these filenames are used
        PATH_TO_CERTIFICATE = gg_install_root + "/" + "thingCert.crt"
        PATH_TO_PRIVATE_KEY = gg_install_root + "/" + "privKey.key"
        PATH_TO_AMAZON_ROOT_CA_1 = gg_install_root  + "/" + root_cert_filename
    else:
        # Manual GGv2 deployment
        #PATH_TO_CERTIFICATE = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-cert.pem"
        #PATH_TO_PRIVATE_KEY = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-privkey.pem"
        PATH_TO_CERTIFICATE = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-certificate.pem.crt"
        PATH_TO_PRIVATE_KEY = gg_install_root + "/certificates/" + gg_endpoint_uuid + "-private.pem.key"
        PATH_TO_AMAZON_ROOT_CA_1 = gg_install_root + "/certificates/" + root_cert_filename

    myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(COMPONENT_CLIENTID)
    myAWSIoTMQTTClient.configureEndpoint(ENDPOINT, 8883)
    myAWSIoTMQTTClient.configureCredentials(PATH_TO_AMAZON_ROOT_CA_1, PATH_TO_PRIVATE_KEY, PATH_TO_CERTIFICATE)
    myAWSIoTMQTTClient.connect()
    return myAWSIoTMQTTClient

def publish_results_to_iotcore(handle,input0,result,result_topic):
    try:
        message = {"input":input0, "result":result}
        handle.publishAsync(result_topic, json.dumps(message), 1, None)
    except:
        e = sys.exc_info()[0]
        print("WindFarm_Optimizer: Exception in publish_results_to_iotcore: " + str(e))

    return

def stop_received(json_input):
    try:
        check = json_input['shutdown']
        if "yes" in check:
            return True
        else:
            return False
    except:
        return False

#### Optimizer: 
def optimal_manipulated_vars(model_name, wind_speed, direction, max_iterations):
    # Admissible bounds for manipulated variables
    bounds = ((0.0, 25.0), (0.0, 25.0))
    
    def objective(x):
        cost = 0.0
        try:
            '''
            Objective function to minimize: Wind Farm
            '''
            # DA: COMMENT OUT: manipulated_vars = pd.DataFrame({"speed": [wind_speed], "direction": [wind_direction],"turbine0_yaw_angle": [x[0]], "turbine1_yaw_angle": [x[1]]})
            predict_input = {"speed":wind_speed,"direction":direction,"turbine0_yaw_angle":0.0,"turbine1_yaw_angle":0.0}

            # DA: call predict()...
            prediction = predict(model_name, predict_input)

            # Deviation from reference 
            if "turbine0_power" and "turbine1_power" and "turbine2_power" and "turbine3_power" in prediction:
                cost = -(prediction["turbine0_power"]+prediction["turbine1_power"]+
                         prediction["turbine2_power"]+prediction["turbine3_power"])
            else:
                print("WindFarm_Optimizer(objective): Model was not loaded: " + model_name + ". Unable to optimize")
        except:
            e = sys.exc_info()[0]
            print("WindFarm_Optimizer: Exception in WindFarm_Optimizer(objective): " + str(e))
            print(traceback.format_exc())

        return cost

    # DA: Cleaning up output for nice logging...
    print("WindFarm_Optimizer: Starting dual_annealing() iterations (Max: " + str(max_iterations) + " iterative predict() calls)...")
    tic=timeit.default_timer()
    result = dual_annealing(objective, bounds=bounds, maxiter=max_iterations)
    toc=timeit.default_timer()
    print("WindFarm_Optimizer: Completed dual_annealing().")
    
    return (result['x'], result['nfev'], -result['fun'], toc-tic)
#### Optimizer

def process_input(client, userdata, message):
    global main_loop_on
    global handle
    global model_name
    global model_unload
    global model_url
    global g_max_iterations
    json_input = {"status":"not started"}

    try:
        # make sure our model is loaded
        if model_unload == False:
            print("WindFarm_Optimizer: process_input(): Loading model: " + model_url)
            bind_to_seam(model_url)

        json_input = json.loads(message.payload.decode())
        if stop_received(json_input):
            main_loop_on = False
        else:
            if model_unload == False:
                print("WindFarm_Optimizer: process_input(): Loading model: " + model_url)
                bind_to_seam(model_url)
            if model_unload == True:
                # Make sure we have valid params
                if "speed" and "direction" in json_input:
                    if isinstance(json_input['speed'], float) or isinstance(json_input['speed'], int):
                        if isinstance(json_input['direction'], float) or isinstance(json_input['direction'], int):
                            # Assign a UUID if one was not provided
                            my_uuid = str(uuid.uuid4())
                            if "uuid" in json_input:
                                my_uuid = json_input['uuid']

                            # If max_iterations provided, use it...
                            if "max_iterations" in json_input and isinstance(json_input['max_iterations'], int):
                                g_max_iterations = json_input['max_iterations']

                            # Immediately send dispatch that optimizer invoked
                            in_progress = {"status": "running", "uuid": my_uuid}
                            print("WindFarm_Optimizer: STARTING: Optimization started: " + str(in_progress) + " to topic: " + result_topic + "...")
                            publish_results_to_iotcore(handle,json_input,in_progress,result_topic)

                            # Invoke the optimizer... LONG WINDED...
                            print("WindFarm_Optimizer: Calling optimal_manipulated_vars() with speed: " + str(json_input['speed']) + " direction: " + str(json_input['direction']) + " Max Iterations: " + str(g_max_iterations))
                            x, nfev, fev, elapsed_secs = optimal_manipulated_vars(model_name, json_input['speed'], json_input['direction'], g_max_iterations)

                            # Format the response
                            final_response = {}
                            final_response['speed'] = json_input['speed']
                            final_response['direction'] = json_input['direction']
                            final_response['turbine0_yaw_angle'] = round(x[0],2)
                            final_response['turbine1_yaw_angle'] = round(x[1],2)
                            final_response['wind_farm_power_kw'] = round(fev,2)
                            final_response['iterations'] = nfev
                            final_response['status'] = "finished"
                            final_response['uuid'] = my_uuid
                            final_response['execute_time_secs'] = round(elapsed_secs,1)

                            # post the results to IoTCore MQTT topic
                            print("WindFarm_Optimizer: FINISHED: Optimizing results: " + str(final_response) + " to topic: " + result_topic + "...")
                            publish_results_to_iotcore(handle,json_input,final_response,result_topic)
                        else:
                            # Incorrect format - direction
                            print("WindFarm_Optimizer: direction supplied in incorrect format. Must be float or integer.")
                            fmt_predict_response = {"status":"error","info":"direction supplied in incorrect format. Must be float or integer"}
                            publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)
                    else:
                        # Incorrect format - speed
                        print("WindFarm_Optimizer: speed supplied in incorrect format. Must be float or integer.")
                        fmt_predict_response = {"status":"error","info":"speed supplied in incorrect format. Must be float or integer"}
                        publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)
                else:
                    # invalid params... ignore
                    print("WindFarm_Optimizer: Invalid parameters supplied. Please check input JSON...must contain speed and direction keys.")
                    fmt_predict_response = {"status":"error","info":"missing speed and/or direction keys in input"}
                    publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)
            else:
                # invalid params... ignore
                print("WindFarm_Optimizer: No models appear loaded.")
                fmt_predict_response = {"status":"error","info":"no models appear loaded"}
                publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)
    except:
            e = sys.exc_info()[0]
            print("WindFarm_Optimizer: Exception in process_input(): " + str(e))
            print(traceback.format_exc())
            json_input = {"status":"in exception"}
            fmt_predict_response = {"status":"exception","info":str(e)}
            publish_results_to_iotcore(handle,json_input,fmt_predict_response,result_topic)

def main_loop(model_name):
    print("WindFarm_Optimizer: Entering main loop()...")
    global main_loop_on
    while main_loop_on == True:
        time.sleep(num_secs)

def bind_to_seam(model_url):
    global model_loaded
    global model_unload

    # See if our model is already loaded
    try: 
        list = list_models()
        for model in list.models:
            print("WindFarm_Optimizer: Found Model: " + model.name)
            if model.name == model_name:
                print("WindFarm_Optimizer: Model: " + model_name + " already loaded. OK")
                model_loaded = False
                model_unload = True
                break
            else:
                print("WindFarm_Optimizer: Model: " + model_name + " NOT loaded. Loading...")
        else:
            print("WindFarm_Optimizer: No models loaded. Loading: " + model_name + "...")
    except:
        print("WindFarm_Optimizer: No models loaded. Loading: " + model_name + "...")

    # Load the model
    if model_loaded == True: 
        print("WindFarm_Optimizer: Loading model: " + model_name + " URL: " + model_url)
        load_model(model_name, model_url)
        model_unload = True

    # List the model
    print("WindFarm_Optimizer: Listing Models:")
    print(list_models())
    list = list_models()
    if hasattr(list, 'models'):
        if len(list.models) > 0:
            print("First Model: " + list.models[0].name)
        else:
            print("No models loaded (OK).")
    else:
        print("No models loaded (grpc down).")

def main():
    global main_loop_on
    global model_name
    global result_topic
    global handle
    global model_url
    global num_tries
    global model_unload

    # Defaults
    parameters = json.loads(os.getenv("AWS_WF_PARAMETERS"))
    model_name = parameters['model_name']
    target_device = parameters['target_device']
    gg_install_root = parameters['gg_install_root']
    gg_endpoint_url = parameters['gg_endpoint_url']
    gg_endpoint_uuid = parameters['gg_endpoint_uuid']
    root_cert_filename = parameters['root_cert_filename']
    result_topic = parameters['result_topic']
    input_topic = parameters['input_topic']
    model_url = gg_install_root + "/work/" + model_name + "-" + target_device + "-component"

    try:
        # Display the configuration
        print("WindFarm_Optimizer: Configuration: Model: " + model_name + " URL: " + model_url)
        print("WindFarm_Optimizer: Socket: " + agent_socket)

        # Bind to SageMaker Edge Agent Manager (seam)
        i = 0
        while model_unload == False and i < num_tries:
            bind_to_seam(model_url)
            time.sleep(num_secs)
            i = i + 1

        # Connect to IoTCore for results publication to MQTT
        print("WindFarm_Optimizer: Connecting to IoTCore for MQTT results publishing...")
        handle = connect_iotcore(gg_install_root,gg_endpoint_uuid,root_cert_filename,gg_endpoint_url)

        # Subscribe to the input topic
        print("WindFarm_Optimizer: Subscribing to input topic: " + input_topic)
        handle.subscribe(input_topic, 1, process_input)

        # Enter the main loop looking for inputs and calling predict()
        main_loop(model_name)

        # unload the model as the event loop is exiting...
        if model_unload == True:
            print("WindFarm_Optimizer: Unloading model: " + model_name)
            unload_model(model_name)

        # Close down the MQTT connection
        print("WindFarm_Optimizer: Main loop closing... disconnecting from AWS...")
        handle.disconnect()
    except:
        e = sys.exc_info()[0]
        print("WindFarm_Optimizer: Exception in main: " + str(e))
        print(traceback.format_exc())
        
    # DONE
    print("WindFarm_Optimizer: DONE! Exiting.")

if __name__ == "__main__":
    main()
