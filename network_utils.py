"""
network_utils.py — Network Scanning and IP Discovery Utilities
==============================================================

Provides helper functions for discovering local IP addresses and scanning
the LAN for reachable devices. Useful for finding the server's IP address
when deploying the federated learning system across multiple machines.

Usage::

    from network_utils import get_local_ip, ping_device, get_network_devices

    # Find this machine's LAN IP
    ip = get_local_ip()

    # Check if server is reachable
    if ping_device("192.168.1.100"):
        print("Server is online")

    # Scan entire subnet
    devices = get_network_devices()
"""

import subprocess
import platform
import socket
import ipaddress
from typing import List, Optional, Tuple
import threading
import queue


def get_local_ip() -> str:
    """Get the primary LAN IP address of the current device.

    Uses a UDP socket connect trick: connecting to an external IP causes the OS
    to select the correct outgoing network interface without actually sending
    any data over the network.

    Returns:
        str: The local IP address (e.g. ``"192.168.1.105"``). Falls back to
            ``"127.0.0.1"`` (loopback) if the address cannot be determined.

    Example::

        ip = get_local_ip()
        print(f"Connect clients to: {ip}:8080")
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))  # Google's public DNS server
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return "127.0.0.1"  # Fallback to localhost if unable to determine IP


def get_network_devices(
    ip_range: Optional[str] = None,
    timeout: float = 1.0
) -> List[Tuple[str, str, str]]:
    """Scan the local subnet for reachable devices using parallel ping.

    Spawns one thread per IP address in the subnet (up to 50 concurrent threads)
    and uses the OS ``ping`` command to check reachability. Threads are daemon
    threads so they don't block process exit.

    Args:
        ip_range (str, optional): CIDR notation of the subnet to scan, e.g.
            ``"192.168.1.0/24"``. If ``None``, automatically detects the local
            subnet from ``get_local_ip()``.
        timeout (float): Seconds to wait for each ping thread to complete.
            Default: ``1.0``.

    Returns:
        list[tuple[str, str, str]]: List of ``(ip, hostname, status)`` tuples
            for every reachable device found. ``hostname`` may be ``"Unknown"``
            if reverse DNS lookup fails. ``status`` is always ``"Online"``.

    Example::

        devices = get_network_devices("192.168.1.0/24")
        for ip, hostname, status in devices:
            print(f"{ip:15s}  {hostname:20s}  {status}")
    """
    if ip_range is None:
        local_ip = get_local_ip()
        network = ".".join(local_ip.split(".")[:-1]) + ".0/24"
    else:
        network = ip_range

    devices = []
    results_queue = queue.Queue()

    def ping_host(ip: str) -> None:
        """Ping a single host and push result to the shared queue if reachable."""
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', '-w', '1000', ip]
            response = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if response.returncode == 0:
                try:
                    hostname = socket.gethostbyaddr(ip)[0]
                except (socket.herror, socket.gaierror):
                    hostname = "Unknown"
                results_queue.put((ip, hostname, "Online"))
        except Exception:
            pass  # Silently skip unreachable or erroring hosts

    threads = []
    for ip in ipaddress.IPv4Network(network, strict=False):
        ip_str = str(ip)
        if ip_str.endswith('.0') or ip_str.endswith('.255'):
            continue  # Skip network address and broadcast address

        t = threading.Thread(target=ping_host, args=(ip_str,))
        t.daemon = True
        threads.append(t)
        t.start()

        # Flush thread batch every 50 to limit concurrency
        if len(threads) >= 50:
            for t in threads:
                t.join(timeout=timeout)
            threads = []

    # Wait for remaining threads
    for t in threads:
        t.join(timeout=timeout)

    # Drain the results queue
    while not results_queue.empty():
        devices.append(results_queue.get())

    return devices


def ping_device(ip: str, count: int = 4) -> bool:
    """Ping a specific IP address and return whether it is reachable.

    Uses the OS ``ping`` command with a 1-second timeout per packet. Works on
    both Windows (``ping -n``) and Unix/macOS (``ping -c``).

    Args:
        ip (str): Target IP address to ping (e.g. ``"192.168.1.100"``).
        count (int): Number of ICMP echo requests to send. Default: ``4``.

    Returns:
        bool: ``True`` if all pings succeeded (return code 0), ``False`` otherwise.

    Example::

        if ping_device("192.168.1.100", count=1):
            print("Server is reachable")
        else:
            print("Server is offline or unreachable")
    """
    try:
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, str(count), '-w', '1000', ip]
        response = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return response.returncode == 0
    except Exception:
        return False


def main() -> None:
    """Demo: print local IP and scan the LAN for reachable devices."""
    print(f"Your local IP address: {get_local_ip()}")

    print("\nScanning local network for devices...")
    devices = get_network_devices()

    if not devices:
        print("No devices found on the network.")
        return

    print("\nFound the following devices:")
    print("-" * 50)
    print(f"{'IP Address':<15} | {'Hostname':<20} | Status")
    print("-" * 50)
    for ip, hostname, status in devices:
        print(f"{ip:<15} | {hostname:<20} | {status}")

    print("\nYou can ping any of these devices using the ping_device() function.")
    print("Example: ping_device('192.168.1.100')")


if __name__ == "__main__":
    main()
