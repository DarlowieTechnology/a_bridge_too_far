from django.shortcuts import render

import logging
import sys


sys.path.append("..")


def index(request):
    # create session key and log per session
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("indexer:" + request.session.session_key)
    logger.info(f"Starting session")

    return render(request, "indexer/index.html", None)


def process(request):

    context = {}

    logger = logging.getLogger("indexer:" + request.session.session_key)

    if request.method == "GET":
        logger.info(f"Serving GET")
        return render(request, "indexer/process.html", context)

    # start POST processing
    logger.info(f"Serving POST")
    return render(request, "indexer/process.html", context)

