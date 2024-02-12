# Built-in imports
import sys
import subprocess as subp
import platform
import argparse
import logging

# Third-party imports
import netfilterqueue

# OpenAPi imports
# import api.core.fileio
# import api.core.dataio




class Net():
  """Class for ip related operations
  """


  @classmethod
  def createIpTables(cls, mode:str="local", queueIndx:int=0) -> int:
    """Create ip tables.

    Args:
      mode (str): Can be local - for testing or remote for attacking.

    Note:
      For local machine testing:
        sudo iptables -I OUTPUT -j NFQUEUE --queue-num 0 
        sudo iptables -I INPUT -j NFQUEUE --queue-num 0 
      For remote machine testing:
        sudo iptables -I FORWARD -j NFQUEUE --queue-num 0

    """
    match mode:
      case "local":
        subp.call(["sudo", "iptables", "-I", "OUTPUT", "-j", "NFQUEUE", "--queue-num", str(queueIndx)])
        subp.call(["sudo", "iptables", "-I", "INPUT", "-j", "NFQUEUE", "--queue-num", str(queueIndx)])
      case "remote":
        subp.call(["sudo", "iptables", "-I", "FORWARD", "-j", "NFQUEUE", "--queue-num", str(queueIndx)])

    return queueIndx


  @classmethod
  def bindIpTables(cls, queueIndx:int, function):
    queue = netfilterqueue.NetfilterQueue()
    queue.bind(queueIndx, function)
    queue.run()


  @classmethod
  def flushIpTables(cls, queueIndx:int=0):
    subp.call(["sudo", "iptables", "--flush"])

