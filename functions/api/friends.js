export async function onRequestGet({ env }) {
  const db = env.DB ?? env.db;

  if (!db) {
    return Response.json({ error: "D1 binding is missing" }, { status: 500 });
  }

  try {
    const { results } = await db
      .prepare(
        `SELECT id, name, url, description, avatar_url
         FROM friend_links
         ORDER BY sort_order ASC, id DESC`,
      )
      .all();

    return Response.json(results ?? [], {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    console.error("Unable to read friend links", error);
    return Response.json({ error: "Unable to load friend links" }, { status: 500 });
  }
}
