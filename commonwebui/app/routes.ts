import {
    type RouteConfig, 
    route,
    index 
} from "@react-router/dev/routes";

export default [
    index("routes/applist.tsx"),
    route("indexer", "routes/indexer.tsx"),
    route("indexer/config", "routes/indexerconfig.tsx"),
    route("query", "routes/query.tsx"),
    route("query/config", "routes/queryconfig.tsx"),    
    route("discovery", "routes/discovery.tsx"),
    route("discovery/config", "routes/discoveryconfig.tsx"),
    route("flexlab", "routes/flexlab.tsx")
] satisfies RouteConfig;
