#!/usr/bin/env python3

import pandas
import os

###################
#
# Extract netflow data from a pcap
# Load csv file into pandas
#
#
##################


def netflow_fields(fields):
    command = ''
    for field in fields:
        command += '-e ' + field + ' '
    return command

def netflow_from_pcap(src_pcap, dst_path, fields, exe_to_run=r'"C:\program files\wireshark\tshark.exe"'):
    print('Creating netflow file from :' + src_pcap)
    print('Creating file : ' + dst_path)
    print('Containing fields : ')
    for field in fields:
        print(field)

    # "C:\program files\wireshark\tshark.exe" -r C:\ai\datasets\ctu13\CTU-13-Dataset\13\botnet-capture-20110815-fast-flux-2.pcap -T fields -e frame.number -e frame.time_epoch -e eth.src -e eth.dst -e\
    # ip.src -e ip.dst -e ip.proto -e tcp.srcport -e udp.srcport -e tcp.dstport -e udp.dstport -e frame.len -E header=y -E separator=, -E quote=n -E occurrence=f
    command = exe_to_run + ' -r ' + src_pcap + ' -T fields '
    command += netflow_fields(fields)
    command += '-E header=y -E separator=, -E quote=n -E occurrence=f >' + dst_path

    print(command)
    os.system(command)


# Example
if __name__ == "__main__":
    # testing
    fields=['frame.number', 'frame.time_epoch', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.proto', 'tcp.srcport', 'udp.srcport', 'tcp.dstport', 'udp.dstport', 'frame.len']

    src_pcap = r'C:\ai\datasets\ctu13\CTU-13-Dataset\13\botnet-capture-20110815-fast-flux-2.pcap'
    dst_file = 'ctu13_13.csv'
    dst_path = os.path.join(os.getcwd(), 'data', dst_file)

    netflow_from_pcap(src_pcap, dst_path, fields)

