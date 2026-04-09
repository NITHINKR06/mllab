import pandas as pd
import math

data={
    'protocol_type': ['tcp','udp','tcp','icmp','tcp','udp','icmp','tcp'],
    'src_bytes'    : [491,146,232,199,420,300,0,500],
    'logged_in'    : [0,0,1,0,1,0,0,1],
    'class'        : ['normal','intrusion','normal','intrusion','normal','intrusion','intrusion','normal']
    }

df=pd.DataFrame(data)

def entropy(column):
    values = column.value_counts()
    total  = len(column)
    ent    = 0
    for v in values:
        p = v / total
        ent -= p*math.log2(p)
    return ent

def information_gain(data, attribute, target):
    total_entropy= entropy(data[target])
    values = data[attribute].unique()
    weighted_entropy = 0
    
    for value in values:
        subset = data[data[attribute] == value]
        weight = len(subset)/ len(data)
        weighted_entropy += weight * entropy(subset[target])
        
        gain = total_entropy - weighted_entropy
        return gain

features = ['protocol_type','src_bytes','logged_in']
target   = 'class'
print("information Gain for each feature:\n")

for feature in features:
    gain = information_gain(df,feature,target)
    print(feature,":",round(gain,3))
    
best_feature = max(features,key=lambda x: information_gain(df,x,target))
print("\n Best Feature for Root Node", best_feature)

def classify(protocol,bytes_sent,login):
    
    if login == 1:
        return "normal"
    else:
        if protocol == "icmp":
            return "intrusion"
        else:
            return "intrusion"
        
        
print("\nTesting new network connection")
protocol = "tcp"
src_bytes = 300
logged_in = 0

prediction = classify(protocol, src_bytes, logged_in)
print("Prediction:", prediction)
    

