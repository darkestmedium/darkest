# Built-in imports
import argparse

# Third-party imports
import scapy.all as scapy




def scan(ip:str):
  arp_request = scapy.ARP(pdst=ip)
  broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
  arpRequestBroadcast = broadcast/arp_request
  listAnswered = scapy.srp(arpRequestBroadcast, timeout=1, verbose=False)[0]
  listClients = []
  for element in listAnswered:
    dictClient = {
      "ip": element[1].psrc,
      "mac": element[1].hwsrc
    }
    listClients.append(dictClient)
  return listClients


def printResults(results:list):
  print("IP\t\t\tMAC Address\n-----------------------------------------")
  [print(f"{client['ip']}\t\t{client['mac']}") for client in results]




def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--target", default="192.168.8.1/24", type=str, help="Target ip range.")
  return parser.parse_args()


if __name__ == "__main__":
  args = syntaxCreator()
  results = scan(args.target)

  printResults(results)