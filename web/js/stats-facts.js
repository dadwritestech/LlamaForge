// Human-scale comparisons for generated tokens. Every number is deliberately
// approximate: token-to-word ratios vary by language/model, book counts vary by
// edition, and route distances vary by road choice.
const WORDS_PER_TOKEN = 0.75;
const ORAL_WORDS_PER_MINUTE = 183;

const BOOKS = [
  ["harry-potter-1", "copy", "copies", "Harry Potter and the Philosopher's Stone", 76_944],
  ["harry-potter-series", "complete", "complete", "Harry Potter series", 1_084_170],
  ["lord-of-the-rings", "copy", "copies", "The Lord of the Rings", 481_103],
  ["war-and-peace", "copy", "copies", "War and Peace", 561_304],
  ["moby-dick", "copy", "copies", "Moby-Dick", 206_052],
  ["pride-and-prejudice", "copy", "copies", "Pride and Prejudice", 121_934],
  ["alice-in-wonderland", "copy", "copies", "Alice's Adventures in Wonderland", 27_170],
  ["huckleberry-finn", "copy", "copies", "Adventures of Huckleberry Finn", 110_997],
  ["jungle-book", "copy", "copies", "The Jungle Book", 51_090],
  ["war-of-the-worlds", "copy", "copies", "The War of the Worlds", 60_556],
  ["wizard-of-oz", "copy", "copies", "The Wonderful Wizard of Oz", 39_074],
  ["hound-baskervilles", "copy", "copies", "The Hound of the Baskervilles", 59_132],
  ["peter-pan", "copy", "copies", "Peter Pan", 47_097],
  ["martin-eden", "copy", "copies", "Martin Eden", 139_281],
  ["fahrenheit-451", "copy", "copies", "Fahrenheit 451", 46_118],
  ["to-kill-a-mockingbird", "copy", "copies", "To Kill a Mockingbird", 100_388],
  ["common-sense", "copy", "copies", "Common Sense", 25_033],
  ["politics-english-language", "copy", "copies", "Politics and the English Language", 5_980],
];

const MOVIES = [
  ["titanic", "Titanic", 195],
  ["oppenheimer", "Oppenheimer", 180],
  ["avengers-endgame", "Avengers: Endgame", 181],
  ["godfather", "The Godfather", 175],
  ["seven-samurai", "Seven Samurai", 207],
  ["spirited-away", "Spirited Away", 125],
  ["dark-knight", "The Dark Knight", 152],
  ["matrix", "The Matrix", 136],
  ["star-wars", "Star Wars", 121],
  ["inception", "Inception", 148],
  ["interstellar", "Interstellar", 169],
  ["dune-part-two", "Dune: Part Two", 166],
  ["shawshank", "The Shawshank Redemption", 142],
  ["jurassic-park", "Jurassic Park", 127],
  ["lotr-extended", "The Lord of the Rings extended trilogy", 683],
];

// Approximate road distances, rounded so the UI does not imply false precision.
const ROUTES = [
  ["munich-frankfurt", "Munich–Frankfurt", 400],
  ["berlin-paris", "Berlin–Paris", 1_050],
  ["london-edinburgh", "London–Edinburgh", 650],
  ["new-york-washington", "New York–Washington, DC", 365],
  ["los-angeles-san-francisco", "Los Angeles–San Francisco", 615],
  ["boston-new-york", "Boston–New York", 350],
  ["toronto-montreal", "Toronto–Montreal", 540],
  ["sydney-melbourne", "Sydney–Melbourne", 880],
  ["tokyo-osaka", "Tokyo–Osaka", 500],
  ["delhi-agra", "Delhi–Agra", 230],
  ["bengaluru-chennai", "Bengaluru–Chennai", 350],
  ["rome-milan", "Rome–Milan", 575],
  ["madrid-barcelona", "Madrid–Barcelona", 620],
  ["lisbon-porto", "Lisbon–Porto", 315],
  ["amsterdam-brussels", "Amsterdam–Brussels", 210],
  ["copenhagen-stockholm", "Copenhagen–Stockholm", 655],
  ["oslo-bergen", "Oslo–Bergen", 465],
  ["cape-town-johannesburg", "Cape Town–Johannesburg", 1_400],
  ["nairobi-mombasa", "Nairobi–Mombasa", 485],
  ["buenos-aires-cordoba", "Buenos Aires–Córdoba", 700],
  ["vienna-prague", "Vienna–Prague", 335],
  ["zurich-geneva", "Zürich–Geneva", 280],
  ["seattle-portland", "Seattle–Portland", 280],
  ["dubai-abu-dhabi", "Dubai–Abu Dhabi", 140],
  ["kuala-lumpur-singapore", "Kuala Lumpur–Singapore", 350],
];

const PACES = [
  ["car", "trip", "trips", "by car", 100],
  ["walk", "journey", "journeys", "at walking pace", 5],
  ["cycle", "journey", "journeys", "at cycling pace", 20],
  ["rail", "journey", "journeys", "at high-speed-rail pace", 250],
];

function rounded(value) {
  const places = value < 10 ? 2 : value < 100 ? 1 : 1;
  return Number(value.toFixed(places));
}

function shown(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 10_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function comparison(id, value, singular, plural, assumption) {
  const amount = rounded(value);
  const unit = Math.abs(amount - 1) < 0.05 ? singular : plural;
  return {id, amount, text: `${shown(amount)} ${unit}`, assumption};
}

export function buildTokenFacts(generatedTokens) {
  const tokens = Math.max(0, Number(generatedTokens) || 0);
  const words = tokens * WORDS_PER_TOKEN;
  const spokenMinutes = words / ORAL_WORDS_PER_MINUTE;
  const common = "English estimate: 0.75 words/token";
  const facts = [];

  for (const [id, singular, plural, title, wordCount] of BOOKS) {
    const one = id === "harry-potter-series" ? `complete ${title}` : `${singular} of ${title}`;
    const many = id === "harry-potter-series" ? `complete ${title}` : `${plural} of ${title}`;
    facts.push(comparison(
      `book:${id}`, words / wordCount, one, many,
      `${common}; ${title}: approximately ${wordCount.toLocaleString()} words`,
    ));
  }
  for (const [id, title, runtime] of MOVIES) {
    facts.push(comparison(
      `movie:${id}`, spokenMinutes / runtime, `back-to-back screening of ${title}`, `back-to-back screenings of ${title}`,
      `${common}; read aloud at 183 words/minute; ${title}: approximately ${runtime} minutes`,
    ));
  }
  for (const [routeId, route, distance] of ROUTES) {
    for (const [paceId, singular, plural, suffix, speed] of PACES) {
      const travelled = spokenMinutes / 60 * speed;
      facts.push(comparison(
        `route:${paceId}:${routeId}`, travelled / distance, `${route} ${singular} ${suffix}`, `${route} ${plural} ${suffix}`,
        `${common}; read aloud at 183 words/minute; ${speed} km/h; ${route}: approximately ${distance.toLocaleString()} km`,
      ));
    }
  }
  return facts;
}

export function createFactRotator({random = Math.random, intervalMs = 15 * 60 * 1000} = {}) {
  let selectedId = null;
  let expiresAt = -Infinity;
  return (generatedTokens, now = Date.now()) => {
    if (!(Number(generatedTokens) > 0)) {
      selectedId = null;
      expiresAt = -Infinity;
      return {id: "empty", amount: 0, text: "Generate some tokens to unlock a real-world comparison", assumption: "Uses generated output tokens only"};
    }
    const catalog = buildTokenFacts(generatedTokens);
    const current = catalog.find(fact => fact.id === selectedId);
    if (current && now < expiresAt) return current;

    // A technically correct "0.00 books" is still a useless comparison. Until
    // a named anchor reaches a readable tenth, show the underlying word scale.
    const eligible = catalog.filter(fact => fact.amount >= 0.1);
    if (!eligible.length) {
      const amount = rounded(Number(generatedTokens) * WORDS_PER_TOKEN);
      return {id: "words", amount, text: `${shown(amount)} English words`, assumption: "English estimate: 0.75 words/token"};
    }
    let next = Math.min(eligible.length - 1, Math.floor(Math.max(0, random()) * eligible.length));
    if (eligible.length > 1 && eligible[next].id === selectedId) {
      next = Math.min(eligible.length - 1, Math.floor(Math.max(0, random()) * eligible.length));
      if (eligible[next].id === selectedId) {
        next = (eligible.findIndex(fact => fact.id === selectedId) + 1) % eligible.length;
      }
    }
    selectedId = eligible[next].id;
    expiresAt = now + intervalMs;
    return eligible[next];
  };
}

export function normalizeVram(payload) {
  if (!payload || !Array.isArray(payload.gpus)) return [];
  return payload.gpus.flatMap((gpu, position) => {
    const total = Number(gpu && gpu.total);
    if (!Number.isFinite(total) || total <= 0) return [];
    const rawUsed = Number(gpu.used);
    const used = Number.isFinite(rawUsed) ? Math.min(total, Math.max(0, rawUsed)) : 0;
    const numeric = key => Number.isFinite(Number(gpu[key])) ? Number(gpu[key]) : null;
    return [{
      index: gpu.index == null ? position : gpu.index,
      name: gpu.name || `GPU ${position}`,
      used,
      total,
      free: total - used,
      util: numeric("util"),
      temp: numeric("temp"),
    }];
  });
}
