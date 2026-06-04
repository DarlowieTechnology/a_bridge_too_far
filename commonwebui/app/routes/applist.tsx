import type { Route } from "./+types/applist";

export async function clientLoader({
  params,
}: Route.ClientLoaderArgs) {
  const response = await fetch(`http://127.0.0.1:8000`);
  //const response = await fetch(`http://127.0.0.1:8000/discovery`);
  //const product = await fetch(`http://127.0.0.1:8000/indexer`);

  if (!response.ok) {
    return "Cannot connect";
  }

  const data = await response.json()
  return data
}

// HydrateFallback is rendered while the client loader is running
export function HydrateFallback() {
  return <div>Loading...</div>;
}

export default function ApplicationList({
    loaderData,
}: Route.ComponentProps) {
    return (
    <main className="flex items-center justify-center pt-16 pb-4">
      <div className="max-w-[300px] w-full space-y-6 px-4">
        <nav className="rounded-3xl border border-gray-200 p-6 dark:border-gray-700 space-y-4">
          <p className="leading-6 text-gray-700 dark:text-gray-200 text-center">
            List of Applications
          </p>
          <ul>
            <li key="Indexer">
              <a
                className="group flex items-center gap-3 self-stretch p-3 leading-normal text-blue-700 hover:underline dark:text-blue-500"
                href="/indexer"
                target="_blank"
                rel="noreferrer"
              >
                "Indexer"
              </a>
            </li>
            <li key="Query">
              <a
                className="group flex items-center gap-3 self-stretch p-3 leading-normal text-blue-700 hover:underline dark:text-blue-500"
                href="/query"
                target="_blank"
                rel="noreferrer"
              >
                "Query"
              </a>
            </li>
            <li key="Discovery">
              <a
                className="group flex items-center gap-3 self-stretch p-3 leading-normal text-blue-700 hover:underline dark:text-blue-500"
                href="/discovery"
                target="_blank"
                rel="noreferrer"
              >
                "Discovery"
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </main>
  );
}
