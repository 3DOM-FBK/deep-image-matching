# Filename: servers.py
# License: LICENSES/LICENSE_UVIC_EPFL
import socket


def is_computecanada():

    hostname = socket.gethostname()

    check = False
    check += hostname.startswith("cedar")
    check += hostname.startswith("gra")
    check += hostname.startswith("cdr")

    return check


def is_vcg_uvic():

    hostname = socket.gethostname()

    check = False
    check += hostname.startswith("kingwood")

    return check


def is_cvlab_epfl():

    hostname = socket.gethostname()

    check = False
    check += hostname.startswith("icc")

    return check
#
# servers.py ends here
