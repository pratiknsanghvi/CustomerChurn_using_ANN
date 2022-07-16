# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:28:07 2022

@author: pratiksanghvi
"""
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
        'secure_connect_bundle': 'D:\Learn & Projects\DeepLearning\Coding\secure-connect-ineuron-insuranceprediction.zip'
}
auth_provider = PlainTextAuthProvider('YcRucwHCMLJnhBzuNRBQDFpM', 'AWmOd8yk_hjQ9pNmxqvDfvyIPtjfoOsrwXM._RGYcoOzz+32KbJp4OpYIhIXR9X,2o1HW+f5HThs_jdPnxJd.xXW5rmDxU9ezC1A4YMSu2L7TTH2HpaO8XNYXfQNeuZA')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
    print(row[0])
else:
    print("An error occurred.")