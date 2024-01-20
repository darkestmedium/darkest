# Built-in imports
import sys
import subprocess
import platform
import argparse
import logging
import re

# Third-party imports
import psutil

# Darkest APi imports




log = logging.getLogger("mac-changer-log")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)




def get_network_interface(mode:str="wireless") -> str:
  """Gets the given network interface.

  Reference:
    for interface_name, interface_addresses in network.items():
      print(f"Interface: {interface_name}")
      for addr in interface_addresses:
        print(f"- Address: {addr.address}, Netmask: {addr.netmask}, Broadcast: {addr.broadcast}")

  """
  network = psutil.net_if_addrs()
  if "lo" in network.keys(): del network["lo"]  # Remove the virtual interface

  for interface_name, _ in network.items():
    match mode:
      case "wired":
        if interface_name.startswith("e"):
          interface = interface_name

      case "wireless":
        if interface_name.startswith("w"):
          interface = interface_name

      case _:
        log.critical("Supported modes are 'wired' and 'wireless'.")
        return

  return interface




def get_mac_address(interface:str) -> str:
  """Returns the mac address for the given network interface.
  """
  output = subprocess.check_output(["ifconfig", args.interface]).decode("utf-8")
  mac_address = re.search(r"\w\w:\w\w:\w\w:\w\w:\w\w:\w\w", output)
  if mac_address:
    return mac_address.group(0)
  else:
    log.info(f"Could not retrieve mac address from {interface}")




def windows(args):
  log.info("Windows support not yet implemented.")


def linux(args):
  mac_curr = get_mac_address(args.interface)

  subprocess.call(["sudo", "ifconfig", args.interface, "down"])
  subprocess.call(["sudo", "ifconfig", args.interface, "hw", "ether", args.mac])
  subprocess.call(["sudo", "ifconfig", args.interface, "up"])

  mac_changed = get_mac_address(args.interface)

  if mac_changed == args.mac:
    log.info(f"Mac address on {args.interface} was changed from {mac_curr} to {mac_changed}")
  else:
    log.warning(f"Mac address on {args.interface} could not be changed.")


def macos(args):
  log.info("MacOS support not yet implemented.")






def syntaxCreator():
  """Creates the command's syntax object and returns it.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--interface", type=str, default=get_network_interface(), help="Name of the network interface.")
  parser.add_argument("-m", "--mac", type=str, default="22:11:22:33:44:55", help="Mac address '00:11:22:33:44:55'.")
  return parser.parse_args()

 
if __name__ == "__main__":
  args = syntaxCreator()

  match platform.system():
    case "Windows":
      windows(args)

    case "Linux":
      linux(args)

    case "Darwin":
      macos(args)

    case _:
      log.info("Other platforms are not supported.")

