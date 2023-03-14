# Import model
from py_utils.utils import *
import torch
import time

# Import OSC
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
import queue
import pickle
import os
import numpy

# Parser for terminal commands
import argparse
parser = argparse.ArgumentParser(description='Monotonic Groove to Drum Generator')
parser.add_argument('--py2pd_port', type=int, default=1123,
                    help='Port for sending messages from python engine to pd (default = 1123)',
                    required=False)
parser.add_argument('--pd2py_port', type=int, default=1415,
                    help='Port for receiving messages sent from pd within the py program (default = 1415)',
                    required=False)
parser.add_argument('--wait', type=float, default=2,
                    help='minimum rate of wait time (in seconds) between two executive generation (default = 2 seconds)',
                    required=False)
parser.add_argument('--show_count', type=bool, default=True,
                    help='prints out the number of sequences generated',
                    required=False)
# parser.add_argument('--model', type=str, default="100",
#                     help='name of the model: (1) light_version: less computationally intensive, or '
#                          '(2) heavy_version: more computationally intensive',
#                     required=False)

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    # ------------------ Load Trained Model  ------------------ #
    model_path = f"trained_torch_models"
    model_name = "100.pth"
    show_count = args.show_count

    groove_transformer_vae = load_model(model_name, model_path)

    voice_thresholds = [0.01 for _ in range(9)]
    voice_max_count_allowed = [16 for _ in range(9)]

    # load per style means/stds of z encodings
    file = open(os.path.join(model_path, "z_means.pkl"), 'rb')
    z_means_dict = pickle.load(file)
    # print(z_means_dict)
    file.close()
    file = open(os.path.join(model_path, "z_stds.pkl"), 'rb')
    z_stds_dict = pickle.load(file)

    print("Available styles: ", list(z_means_dict.keys()))

    ############### GLOBAL VARIABLES ###############
    groovePattern_A_z = numpy.zeros(128)
    def regeneratePatternA_FromStyleNum(styleNum):
        if styleNum >= 0 and styleNum <= 11:
            styleName = list(z_means_dict.keys())[styleNum]
            global groovePattern_A_z
            groovePattern_A_z = get_random_sample_from_style_noDrumGeneration(style_= styleName,
                                    z_means_dict = z_means_dict,
                                    z_stds_dict = z_stds_dict,
                                    scale_means_factor = 1.0, scale_stds_factor = 1.0)
            # print(f'Generated random groove pattern A out of style {styleName}')
            # print(groovePattern_A_z)
    regeneratePatternA_FromStyleNum(0)

    groovePattern_B_z = numpy.zeros(128)
    def regeneratePatternB_FromStyleNum(styleNum):
        if styleNum >= 0 and styleNum <= 11:
            styleName = list(z_means_dict.keys())[styleNum]
            global groovePattern_B_z
            groovePattern_B_z = get_random_sample_from_style_noDrumGeneration(style_ = styleName,
                                        z_means_dict = z_means_dict,
                                        z_stds_dict = z_stds_dict,
                                        scale_means_factor = 1.0, scale_stds_factor = 1.0)
            # print(f'Generated random groove pattern B out of style {styleName}')
            # print(groovePattern_B_z)
    regeneratePatternB_FromStyleNum(1)

    # TEST interpolation
    '''
    patA = [0., 0., 0., 0., 0.]
    patB = [1., 1., 1., 1., 1.]
    print(get_interpolated_z_from_zs(patA, patB, 0.4))
    '''

    # at PD patch startup, the slider is all the way to the left
    interpolatedPattern_z = groovePattern_A_z
    slider1_LastInterpValue = 0.
    slider2_LastInterpValue = 0.
    slider3_LastInterpValue = 0.
    slider4_LastInterpValue = 0.

    loadPatternA_Path = str()
    savePatternA_Path = str()
    loadPatternB_Path = str()
    savePatternB_Path = str()

    slider1_lower_zInterp_Bound = 0
    slider1_upper_zInterp_Bound = 31
    slider2_lower_zInterp_Bound = 32
    slider2_upper_zInterp_Bound = 63
    slider3_lower_zInterp_Bound = 64
    slider3_upper_zInterp_Bound = 95
    slider4_lower_zInterp_Bound = 96
    slider4_upper_zInterp_Bound = 127

    def loadPatternA(picklePath):
        file = open(picklePath, 'rb')
        global groovePattern_A_z
        groovePattern_A_z = pickle.load(file)

    def savePatternA(picklePath):
        file = open(picklePath, 'wb')
        global groovePattern_A_z
        pickle.dump(groovePattern_A_z, file)

    def loadPatternB(picklePath):
        file = open(picklePath, 'rb')
        global groovePattern_B_z
        groovePattern_B_z = pickle.load(file)

    def savePatternB(picklePath):
        file = open(picklePath, 'wb')
        global groovePattern_B_z
        pickle.dump(groovePattern_B_z, file)

    (h_new, v_new, o_new) = (torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)))

    # ------  Create an empty an empty torch tensor
    input_tensor = torch.zeros((1, 32, 3))

    # ------  Create an empty h, v, o tuple for previously generated events to avoid duplicate messages
    (h_old, v_old, o_old) = (torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)), torch.zeros((1, 32, 9)))

    # set the minimum time needed between generations
    min_wait_time_btn_gens = args.wait


    # -----------------------------------------------------

    # ------------------ OSC ips / ports ------------------ #
    # connection parameters
    ip = "127.0.0.1"
    receiving_from_pd_port = args.pd2py_port
    sending_to_pd_port = args.py2pd_port
    message_queue = queue.Queue()
    # ----------------------------------------------------------

    # ------------------ OSC Receiver from Pd ------------------ #
    # create an instance of the osc_sender class above
    py_to_pd_OscSender = SimpleUDPClient(ip, sending_to_pd_port)
    # ---------------------------------------------------------- #

    def process_message_from_queue(address, args):
        print("Address; ".format(address))
        print("Args; ".format(args))
              
        if "VelutimeIndex" in address:
            input_tensor[:, int(args[2]), 0] = 1 if args[0] > 0 else 0  # set hit
            input_tensor[:, int(args[2]), 1] = args[0] / 127  # set velocity
            input_tensor[:, int(args[2]), 2] = args[1]  # set utiming
        elif "threshold" in address:
            voice_thresholds[int(address.split("/")[-1])] = 1-args[0]
        elif "max_count" in address:
            voice_max_count_allowed[int(address.split("/")[-1])] = int(args[0])
        elif "regenerate" in address:
            pass
        elif "time_between_generations" in address:
            global min_wait_time_btn_gens
            min_wait_time_btn_gens = args[0]
        elif "interp_slider_1" in address:
            # print("interp_slider_1 received")
            global slider1_LastInterpValue
            slider1_LastInterpValue = args[0] 
        elif "interp_slider_2" in address:
            global slider2_LastInterpValue
            slider2_LastInterpValue = args[0] 
            # print("interp_slider_2 received")
        elif "interp_slider_3" in address:
            global slider3_LastInterpValue
            slider3_LastInterpValue = args[0] 
            # print("interp_slider_3 received")
        elif "interp_slider_4" in address:
            global slider4_LastInterpValue
            slider4_LastInterpValue = args[0] 
            # print("interp_slider_4 received")
        elif "gen_new_patternA_with_style" in address:
            print("gen_new_patternA_with_style received")
            regeneratePatternA_FromStyleNum(int(args[0]))
        elif "gen_new_patternB_with_style" in address:
            print("gen_new_patternB_with_style received")
            regeneratePatternB_FromStyleNum(int(args[0]))
        elif "load_patternA_path" in address:
            print("load_patternA_path received")
            # print(args[0])
            loadPatternA(args[0])
        elif "save_patternA_path" in address:
            print("save_patternA_path received")
            # print(args[0])
            savePatternA(args[0])
        elif "load_patternB_path" in address:
            print("load_patternB_path received")
            # print(args[0])
            loadPatternB(args[0])
        elif "save_patternB_path" in address:
            print("save_patternB_path received")
            # print(args[0])
            savePatternB(args[0])
        elif "slider1_lower_zInterp_Bound" in address:
            global slider1_lower_zInterp_Bound
            slider1_lower_zInterp_Bound = int(args[0])
            # print(slider1_lower_zInterp_Bound)
        elif "slider1_upper_zInterp_Bound" in address:
            global slider1_upper_zInterp_Bound
            slider1_upper_zInterp_Bound = int(args[0])
            # print(slider1_upper_zInterp_Bound)
        elif "slider2_lower_zInterp_Bound" in address:
            global slider2_lower_zInterp_Bound
            slider2_lower_zInterp_Bound = int(args[0])
            # print(slider2_lower_zInterp_Bound)
        elif "slider2_upper_zInterp_Bound" in address:
            global slider2_upper_zInterp_Bound
            slider2_upper_zInterp_Bound = int(args[0])
            # print(slider2_upper_zInterp_Bound)
        elif "slider3_lower_zInterp_Bound" in address:
            global slider3_lower_zInterp_Bound
            slider3_lower_zInterp_Bound = int(args[0])
            # print(slider3_lower_zInterp_Bound)
        elif "slider3_upper_zInterp_Bound" in address:
            global slider3_upper_zInterp_Bound
            slider3_upper_zInterp_Bound = int(args[0])
            # print(slider3_upper_zInterp_Bound)
        elif "slider4_lower_zInterp_Bound" in address:
            global slider4_lower_zInterp_Bound
            slider4_lower_zInterp_Bound = int(args[0])
            # print(slider4_lower_zInterp_Bound)
        elif "slider4_upper_zInterp_Bound" in address:
            global slider4_upper_zInterp_Bound
            slider4_upper_zInterp_Bound = int(args[0])
            # print(slider4_upper_zInterp_Bound)
        else:
            print ("Unknown Message Received, address {}, value {}".format(address, args))
    # python-osc method for establishing the UDP communication with pd
    server = OscMessageReceiver(ip, receiving_from_pd_port, message_queue=message_queue)
    server.start()

    # ---------------------------------------------------------- #

    def recalculate_z(sliderNum, interpValue):
        # print(f"recalculate_z triggered, patterns size; {groovePattern_A_z.shape, groovePattern_B_z.shape}")
        startIndex = 0
        endIndex = 0
        if sliderNum == 1:
            global slider1_lower_zInterp_Bound
            global slider1_upper_zInterp_Bound
            startIndex = slider1_lower_zInterp_Bound
            endIndex = slider1_upper_zInterp_Bound
        elif sliderNum == 2:
            global slider2_lower_zInterp_Bound
            global slider2_upper_zInterp_Bound
            startIndex = slider2_lower_zInterp_Bound
            endIndex = slider2_upper_zInterp_Bound
        elif sliderNum == 3:
            global slider3_lower_zInterp_Bound
            global slider3_upper_zInterp_Bound
            startIndex = slider3_lower_zInterp_Bound
            endIndex = slider3_upper_zInterp_Bound
        elif sliderNum == 4:
            global slider4_lower_zInterp_Bound
            global slider4_upper_zInterp_Bound
            startIndex = slider4_lower_zInterp_Bound
            endIndex = slider4_upper_zInterp_Bound
        # print(f'Interpolated pattern before slider change; {interpolatedPattern_z}')
        interpolatedPattern_z[startIndex:endIndex] = get_interpolated_z_from_zs(groovePattern_A_z[startIndex:endIndex], groovePattern_B_z[startIndex:endIndex], interpValue)
        # print(f'Interpolated pattern after slider change; {interpolatedPattern_z}')
        # return         

    # ------------------ NOTE GENERATION  ------------------ #
    # drum_voice_pitch_map = {"kick": 36, 'snare': 38, 'tom-1': 47, 'tom-2': 42, 'chat': 64, 'ohat': 63}
    # drum_voices = list(drum_voice_pitch_map.keys())
    
    number_of_generations = 0
    count = 0
    while (1):
        address, args = message_queue.get()
        # print('args', args)
        process_message_from_queue(address, args)

        # only generate new pattern when there isnt any other osc messages backed up for processing in the message_queue
        if (message_queue.qsize() == 0):

            # ----------------------------------------------------------------------------------------------- #
            # ----------------------------------------------------------------------------------------------- #
            # EITHER GENERATE USING GROOVE OR GENERATE A RANDOM PATTERN

            if "interp_slider" in address:
                if "interp_slider_1" in address:
                    recalculate_z(1, args[0])
                elif "interp_slider_2" in address:
                    recalculate_z(2, args[0])
                elif "interp_slider_3" in address:
                    recalculate_z(3, args[0])
                elif "interp_slider_4" in address:
                    recalculate_z(4, args[0])
                # print(interpolatedPattern_z)
                h_new , v_new , o_new = decode_z_into_drums(groove_transformer_vae, interpolatedPattern_z, voice_thresholds, voice_max_count_allowed)
                osc_messages_to_send = get_new_drum_osc_msgs((h_new, v_new, o_new))
                number_of_generations += 1
                # First clear generations on pd by sending a message
                py_to_pd_OscSender.send_message("/reset_table", 1)
                # Then send over generated notes one at a time
                for (address, h_v_ix_tuple) in osc_messages_to_send:
                    py_to_pd_OscSender.send_message(address, h_v_ix_tuple)
                if show_count:
                    print("Generation #", count)
                # Message pd that sent is over by sending the counter value for number of generations
                # used so to take snapshots in pd
                py_to_pd_OscSender.send_message("/generation_count", count)
                count += 1
            # or  "slider1_lower_zInterp_Bound" in address or  "slider1_upper_zInterp_Bound" in address or  "slider2_lower_zInterp_Bound" in address or  "slider2_upper_zInterp_Bound" in address or  "slider3_lower_zInterp_Bound" in address or  "slider3_upper_zInterp_Bound" in address or  "slider4_lower_zInterp_Bound" in address or  "slider4_upper_zInterp_Bound" in address
            elif "gen_new_patternA_with_style" in address or  "gen_new_patternB_with_style" in address or "load_patternA_path" in address or  "load_patternB_path" in address:
                recalculate_z(1, slider1_LastInterpValue)
                recalculate_z(2, slider2_LastInterpValue)
                recalculate_z(3, slider3_LastInterpValue)
                recalculate_z(4, slider4_LastInterpValue)
                h_new , v_new , o_new = decode_z_into_drums(groove_transformer_vae, interpolatedPattern_z, voice_thresholds, voice_max_count_allowed)
                osc_messages_to_send = get_new_drum_osc_msgs((h_new, v_new, o_new))
                number_of_generations += 1
                # First clear generations on pd by sending a message
                py_to_pd_OscSender.send_message("/reset_table", 1)
                # Then send over generated notes one at a time
                for (address, h_v_ix_tuple) in osc_messages_to_send:
                    py_to_pd_OscSender.send_message(address, h_v_ix_tuple)
                if show_count:
                    print("Generation #", count)
                # Message pd that sent is over by sending the counter value for number of generations
                # used so to take snapshots in pd
                py_to_pd_OscSender.send_message("/generation_count", count)
                count += 1
            else:
                print('not our slider')
                
            time.sleep(min_wait_time_btn_gens)