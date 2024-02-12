# Built-in imports
import argparse

# Third-party imports
import netfilterqueue




def processPacket(packet):
  print(packet)
  packet.accept()




# Local machine
# sudo iptables -I OUTPUT -j NFQUEUE --queue-num 0 
# sudo iptables -I INPUT -j NFQUEUE --queue-num 0 
# Remote machine
# sudo iptables -I FORWARD -j NFQUEUE --queue-num 0
queue = netfilterqueue.NetfilterQueue()
queue.bind(0, processPacket)
queue.run()
# sudo iptables --flush
