from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl


# local
from common import ConfigCollection
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
        return app.state.INDEXER
    else:
        configCollection = ConfigCollection()
        configCollection.configure()
        indexerWorkflow = IndexerWorkflow()
        indexerWorkflow.configure(configCollection)
        return indexerWorkflow


@app.get("/indexer/config")
async def indexerconfiguration( updatedWorkflow : Dict[str, Any] ):
    if hasattr(app.state, "INDEXER"):
        indexerWorkflow = app.state.INDEXER
        if not indexerWorkflow.needsUpdate(updatedWorkflow):
            return indexerWorkflow
    configCollection = ConfigCollection()
    configCollection.configure()
    configCollection.update(updatedWorkflow)
    indexerWorkflow = IndexerWorkflow()
    indexerWorkflow.configure(configCollection)
    app.state.QUERY = indexerWorkflow
    return indexerWorkflow


@app.get("/query")
async def query():
    if hasattr(app.state, "QUERY"):
        return app.state.QUERY
    else:
        configCollection = ConfigCollection()
        configCollection.configure()
        queryWorkflow = QueryWorkflow()
        queryWorkflow.configure(configCollection)
        app.state.QUERY = queryWorkflow
        return queryWorkflow
 
@app.get("/query/config")
async def queryconfiguration( updatedWorkflow : Dict[str, Any] ):
    if hasattr(app.state, "QUERY"):
        queryWorkflow = app.state.QUERY
        if not queryWorkflow.needsUpdate(updatedWorkflow):
            return queryWorkflow
    configCollection = ConfigCollection()
    configCollection.configure()
    configCollection.update(updatedWorkflow)
    queryWorkflow = QueryWorkflow()
    queryWorkflow.configure(configCollection)
    app.state.QUERY = queryWorkflow
    return queryWorkflow


@app.get("/discovery")
async def discovery():
    if hasattr(app.state, "DISCOVERY"):
        return app.state.DISCOVERY
    else:
        configCollection = ConfigCollection()
        configCollection.configure()
        discoveryWorkflow = DiscoveryWorkflow()
        discoveryWorkflow.configure(configCollection)
        app.state.DISCOVERY = discoveryWorkflow
        return discoveryWorkflow

@app.post("/discovery/config")
async def discoveryconfiguration( updatedWorkflow : Dict[str, Any] ):
    if hasattr(app.state, "DISCOVERY"):
        discoveryWorkflow = app.state.DISCOVERY
        if not discoveryWorkflow.needsUpdate(updatedWorkflow):
            return discoveryWorkflow
    configCollection = ConfigCollection()
    configCollection.configure()
    configCollection.update(updatedWorkflow)
    discoveryWorkflow = DiscoveryWorkflow()
    discoveryWorkflow.configure(configCollection)
    app.state.DISCOVERY = discoveryWorkflow
    return discoveryWorkflow