// Every bot is named adjective_noun (e.g. brisk_lagoon); its avatar comes from the noun.
const NOUN_EMOJI: Record<string, string> = {
  lagoon: "🌊", badger: "🦡", cactus: "🌵", dune: "🏜️", ember: "🔥", falcon: "🦅", glacier: "🧊", heron: "🐦", island: "🏝️",
  jackal: "🐺", kelp: "🌿", lynx: "🐱", meadow: "🌼", newt: "🦎", orchid: "🌸", pebble: "🪨", quartz: "💎", raven: "🐦‍⬛",
  spruce: "🌲", tundra: "❄️", otter: "🦦", willow: "🍃", yak: "🐃", zephyr: "🌬️",
};

export function avatar(name: string): string {
  if (name === "you" || name === "You") return "😎";
  return NOUN_EMOJI[name.split("_").pop() ?? ""] ?? "🤖";
}

/** "brisk_lagoon" -> "Brisk Lagoon" */
export function prettyName(name: string): string {
  if (name === "you") return "You";
  return name.split("_").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}
