import { defineCollection, z } from "astro:content";

const postsCollection = defineCollection({
	schema: z.object({
		title: z.string(),
		published: z.preprocess((val) => {
			// If frontmatter provides a string, try to parse it to a Date.
			if (typeof val === "string") {
				const d = new Date(val);
				return Number.isNaN(d.getTime()) ? new Date('2000-01-01') : d;
			}
			// If already a Date, keep it.
			if (val instanceof Date) return val;
			// Fallback to 2000-01-01 for undefined or other types.
			return new Date('2000-01-01');
		}, z.date()),
		updated: z.date().optional(),
		draft: z.boolean().optional().default(false),
		description: z.string().optional().nullable().transform((v) => (v == null ? "" : v)),
		image: z.string().optional().default(""),
		tags: z.array(z.string()).optional().default([]),
		category: z.string().optional().nullable().default(null),
		lang: z.string().optional().default(""),

		/* For internal use */
		prevTitle: z.string().default(""),
		prevSlug: z.string().default(""),
		nextTitle: z.string().default(""),
		nextSlug: z.string().default(""),
	}),
});
const specCollection = defineCollection({
	schema: z.object({}),
});
export const collections = {
	posts: postsCollection,
	spec: specCollection,
};
