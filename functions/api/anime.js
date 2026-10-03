export async function onRequestGet({ env }) {
  const db = env.DB ?? env.db;

  if (!db) {
    return Response.json({ error: "D1 binding is missing" }, { status: 500 });
  }

  try {
    const { results } = await db
      .prepare(
        `SELECT id, title, release_year, score, review, cover_url
         FROM anime
         ORDER BY sort_order ASC, id DESC`,
      )
      .all();

    return Response.json(results ?? [], {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    console.error("Unable to read anime records", error);
    return Response.json({ error: "Unable to load anime records" }, { status: 500 });
  }
}
