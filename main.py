from typing import Dict, Any
import json
from fastapi import FastAPI
from fastapi import Body
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from pprint import pprint

# local
from common import ConfigCollection, DebugUtils
from indexer_workflow import IndexerWorkflow
from query_workflow import QueryWorkflow
from discovery_workflow import DiscoveryWorkflow


app = FastAPI()

origins = [
    "http://localhost",
    "https://localhost",
    "http://localhost:5173",        # React Router web server
    "https://localhost:5173",       # React Router web server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FAPIAppCollection(BaseModel):
    indexer : HttpUrl = Field(default = HttpUrl("http://127.0.0.1:8000/indexer"), description="Indexer URL")
    query : HttpUrl = Field(default = HttpUrl("http://127.0.0.1:8000/query"), description="Query URL")
    discovery : HttpUrl = Field(default = HttpUrl("http://127.0.0.1:8000/discovery"), description="Discovery URL")


@app.get("/")
async def read_root():
    fapiAppCollection = FAPIAppCollection()
    return fapiAppCollection


@app.get("/indexer")
async def indexer():
    if hasattr(app.state, "INDEXER"):
#        DebugUtils.logPydanticObject(app.state.INDEXER, "INDEXER FROM CACHE")
        return app.state.INDEXER
    else:
        configCollection = ConfigCollection()
        configCollection.configure()
        indexerWorkflow = IndexerWorkflow()
        indexerWorkflow.configure(configCollection)
        app.state.INDEXER = indexerWorkflow
#        DebugUtils.logPydanticObject(indexerWorkflow, "INDEXER CREATED NEW")
        return indexerWorkflow


@app.post("/indexer/config")
async def indexer_configuration( request: Request ):
    body = await request.body()
    data_dict = json.loads(body.decode('utf-8'))
    configCollection = ConfigCollection()
    configCollection.configure()
    configCollection.update(data_dict)
    indexerWorkflow = IndexerWorkflow()
    indexerWorkflow.configure(configCollection)
    app.state.INDEXER = indexerWorkflow
#    DebugUtils.logPydanticObject(indexerWorkflow, "INDEXER UPDATED STATE")
    return indexerWorkflow


@app.get("/query")
async def query():
    if hasattr(app.state, "QUERY"):
#        DebugUtils.logPydanticObject(app.state.QUERY, "QUERY FROM CACHE")
        return app.state.QUERY
    else:
        configCollection = ConfigCollection()
        configCollection.configure()
        queryWorkflow = QueryWorkflow()
        queryWorkflow.configure(configCollection)
        app.state.QUERY = queryWorkflow
#        DebugUtils.logPydanticObject(queryWorkflow, "QUERY CREATED NEW")
        return queryWorkflow
 
@app.post("/query/config")
async def query_configuration( request: Request ):
    body = await request.body()
    data_dict = json.loads(body.decode('utf-8'))
    configCollection = ConfigCollection()
    configCollection.configure()
    configCollection.update(data_dict)
    queryWorkflow = QueryWorkflow()
    queryWorkflow.configure(configCollection)
    app.state.QUERY = queryWorkflow
#    DebugUtils.logPydanticObject(queryWorkflow, "QUERY UPDATED STATE")
    return queryWorkflow


@app.get("/discovery")
async def discovery():
    if hasattr(app.state, "DISCOVERY"):
#        DebugUtils.logPydanticObject(app.state.DISCOVERY, "DISCOVERY FROM CACHE")
        return app.state.DISCOVERY
    else:
        configCollection = ConfigCollection()
        configCollection.configure()
        discoveryWorkflow = DiscoveryWorkflow()
        discoveryWorkflow.configure(configCollection)
        app.state.DISCOVERY = discoveryWorkflow
#        DebugUtils.logPydanticObject(discoveryWorkflow, "DISCOVERY CREATED NEW")
        return discoveryWorkflow


@app.post("/discovery/config")
async def discovery_configuration( request: Request ):
    body = await request.body()
    data_dict = json.loads(body.decode('utf-8'))
    configCollection = ConfigCollection()
    configCollection.configure()
    configCollection.update(data_dict)
    discoveryWorkflow = DiscoveryWorkflow()
    discoveryWorkflow.configure(configCollection)
    app.state.DISCOVERY = discoveryWorkflow
#    DebugUtils.logPydanticObject(discoveryWorkflow, "DISCOVERY UPDATED STATE")
    return discoveryWorkflow