export async function onRequestGet({ env }) {
  const db = env.DB ?? env.db;

  if (!db) {
    return Response.json({ error: "D1 binding is missing" }, { status: 500 });
  }

  try {
    const { results } = await db
      .prepare(
        `SELECT id, title, category, description, url
         FROM japanese_resources
         ORDER BY sort_order ASC, id DESC`,
      )
      .all();

    return Response.json(results ?? [], {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    console.error("Unable to read Japanese resources", error);
    return Response.json({ error: "Unable to load Japanese resources" }, { status: 500 });
  }
}
