import aliasesJson from './assets/data/aliases.json';

export class NameCoref {
  private honorifics: Record<string, number>;
  private aliases: Record<string, Record<string, number>>;

  constructor() {
    this.honorifics = {
      mr: 1,
      'mr.': 1,
      mrs: 1,
      'mrs.': 1,
      miss: 1,
      uncle: 1,
      aunt: 1,
      lady: 1,
      lord: 1,
      monsieur: 1,
      master: 1,
      mistress: 1,
    };

    this.aliases = {};

    for (const entry of aliasesJson) {
        // each entry expected to be [canonical, ...nicknames]
        const canonical = String(entry[0]);
        const nicknames = entry.slice(1).map(String);
        for (const nickname of nicknames) {
        const key = nickname.toLowerCase();
        if (!(key in this.aliases)) this.aliases[key] = {};
        this.aliases[key][canonical.toLowerCase()] = 1;
        }
    }
  }

  private get_variants(parts: string[]): Record<string, number> {
    const variants: Record<string, number> = {};
    const n = parts.length;
    for (let i = 0; i < n; i++) {
      if (!(parts[i].toLowerCase() in this.honorifics)) {
        variants[parts[i]] = 1;
      }
      for (let j = i + 1; j < n; j++) {
        variants[`${parts[i]} ${parts[j]}`] = 1;
        for (let k = j + 1; k < n; k++) {
          variants[`${parts[i]} ${parts[j]} ${parts[k]}`] = 1;
          for (let l = k + 1; l < n; l++) {
            variants[`${parts[i]} ${parts[j]} ${parts[k]} ${parts[l]}`] = 1;
            for (let m = l + 1; m < n; m++) {
              variants[`${parts[i]} ${parts[j]} ${parts[k]} ${parts[l]} ${parts[m]}`] = 1;
              for (let o = m + 1; o < n; o++) {
                variants[`${parts[i]} ${parts[j]} ${parts[k]} ${parts[l]} ${parts[m]} ${parts[o]}`] = 1;
              }
            }
          }
        }
      }
    }
    return variants;
  }

  private get_canonical(name_tokens: string[]): string[][] {
    const name = name_tokens.join(' ').toLowerCase();
    if (name in this.aliases) {
      const vals: string[][] = [];
      for (const can in this.aliases[name]) {
        vals.push(can.split(' '));
      }
      return vals;
    }

    const parts: string[][] = [];
    for (const tok of name_tokens) {
      const key = tok.toLowerCase();
      if (key in this.aliases) parts.push(Object.keys(this.aliases[key]));
      else parts.push([tok]);
    }

    const canonicals: string[][] = [];
    // cartesian product
    const recurse = (i: number, acc: string[]) => {
      if (i === parts.length) {
        canonicals.push(acc.slice());
        return;
      }
      for (const v of parts[i]) {
        acc.push(v);
        recurse(i + 1, acc);
        acc.pop();
      }
    };
    recurse(0, []);
    return canonicals;
  }

  public name_cluster(
    entities: string[][],
    is_named: number[],
    existing_refs: number[] = []
  ): number[] {
    const uniq: Record<string, number> = {};
    for (let i = 0; i < is_named.length; i++) {
      const val = is_named[i];
      if (val === 1) {
        if (entities[i].length < 10) {
          const name = entities[i].join(' ').toLowerCase();
          if (name !== '') uniq[name] = (uniq[name] || 0) + 1;
        }
      }
    }

    const subsets: Record<string, number> = {};
    for (const name1 in uniq) {
      const canonicals1 = this.get_canonical(name1.split(' '));
      for (const canonical1 of canonicals1) {
        const name1set = new Set(canonical1);
        for (const name2 in uniq) {
          if (name1 === name2) continue;
          const canonicals = this.get_canonical(name2.split(' '));
          for (const canonical of canonicals) {
            const name2set = new Set(canonical);
            if ([...name1set].join(' ') === [...name2set].join(' ')) continue;
            let isSuperset = true;
            for (const el of name2set) if (!name1set.has(el)) { isSuperset = false; break; }
            if (isSuperset) subsets[name2] = 1;
          }
        }
      }
    }

    const name_subpart_index: Record<string, Record<string, number>> = {};
    for (const name in uniq) {
      if (name in subsets) continue;
      const canonicals = this.get_canonical(name.split(' '));
      for (const canonical of canonicals) {
        const variants = this.get_variants(canonical);
        for (const v in variants) {
          if (!(v in name_subpart_index)) name_subpart_index[v] = {};
          name_subpart_index[v][name] = 1;
        }
      }
    }

    const charids: Record<string, number> = {};
    let max_id = 1;
    if (existing_refs.length > 0) {
      max_id = Math.max(...existing_refs) + 1;
    }

    const lastSeen: Record<string, number> = {};
    const refs: number[] = [];
    for (let i = 0; i < is_named.length; i++) {
      if ((existing_refs[i] ?? -1) !== -1) {
        refs.push(existing_refs[i]);
        continue;
      }

      const val = is_named[i];
      if (val === 1) {
        const canonicals = this.get_canonical(entities[i]);
        const name = entities[i].join(' ').toLowerCase();
        let top: string | null = null;
        let max_score = 0;
        for (const canonical of canonicals) {
          const canonical_name = canonical.join(' ').toLowerCase();
          if (canonical_name in name_subpart_index) {
            for (const entity in name_subpart_index[canonical_name]) {
              const score = (uniq[entity] || 0) + (lastSeen[entity] || 0);
              if (score > max_score) {
                max_score = score;
                top = entity;
              }
            }
          }
        }

        if (top !== null) {
          lastSeen[top] = i;
          if (!(top in charids)) {
            charids[top] = max_id;
            max_id += 1;
          }
          refs.push(charids[top]);
        } else {
          refs.push(-1);
        }
      } else {
        refs.push(-1);
      }
    }
    return refs;
  }

  private calc_overlap(small: Record<string, number>, big: Record<string, number>): number {
    let overlap = 0;
    let smallSum = 0;
    for (const name in small) {
      smallSum += small[name];
      if (name in big) overlap += small[name];
    }
    if (smallSum === 0) return 0;
    return overlap / smallSum;
  }

  public cluster_identical_propers(entities: Array<[number, number, string, string]>, refs: number[]): number[] {
    let max_id = 1;
    if (refs.length > 0) max_id = Math.max(...refs) + 1;
    const names: Record<string, number> = {};
    for (let idx = 0; idx < entities.length; idx++) {
      const [, , full_cat, name] = entities[idx];
      const [prop, cat] = full_cat.split('_');
      if (prop === 'PROP' && cat !== 'PER') {
        const n = name.toLowerCase();
        const key = `${n}_${prop}_${cat}`;
        if (!(key in names)) {
          names[key] = max_id;
          max_id += 1;
        }
        refs[idx] = names[key];
      }
    }
    return refs;
  }

  public cluster_noms(entities: Array<[number, number, string, string]>, refs: number[]): number[] {
    const names: Record<string, number> = {};
    const mapper: Record<number, number> = {};
    for (let idx = 0; idx < entities.length; idx++) {
      const [, , cat, name] = entities[idx];
      const prop = cat.split('_')[0];
      if (prop === 'NOM') {
        const n = name.toLowerCase();
        if (!(n in names)) names[n] = refs[idx];
        else mapper[refs[idx]] = names[n];
      }
    }

    for (let idx = 0; idx < refs.length; idx++) {
      const ref = refs[idx];
      if (ref in mapper) refs[idx] = mapper[ref];
    }
    return refs;
  }

  public cluster_narrator(entities: Array<[number, number, string, string]>, in_quotes: number[], tokens: Array<{ text: string }>): number[] {
    const narrator_pronouns = new Set(['i', 'me', 'my', 'myself']);
    const refs: number[] = [];
    for (let idx = 0; idx < entities.length; idx++) {
      const [, , , name] = entities[idx];
      if (in_quotes[idx] === 0 && narrator_pronouns.has(name.toLowerCase())) refs.push(0);
      else refs.push(-1);
    }
    return refs;
  }

  public cluster_only_nouns(
    entities: Array<[number, number, string, string]>,
    refs: number[],
    tokens: Array<{ text: string; pos?: string | null }>
  ): number[] {
    const hon_mapper: Record<string, string> = {
      mister: 'mr.',
      'mr.': 'mr.',
      mr: 'mr.',
      mistah: 'mr.',
      mastah: 'mr.',
      master: 'mr.',
      miss: 'miss',
      'ms.': 'miss',
      ms: 'miss',
      missus: 'miss',
      mistress: 'miss',
      'mrs.': 'mrs.',
      mrs: 'mrs.',
    };

    const map_honorifics = (term: string) => {
      const t = term.toLowerCase();
      return t in hon_mapper ? hon_mapper[t] : null;
    };

    const is_named: number[] = [];
    const entity_names: string[][] = [];

    for (const [start, end, cat, text] of entities) {
      const [ner_prop, ner_type] = cat.split('_');
      if (ner_prop === 'PROP' && ner_type === 'PER') is_named.push(1);
      else is_named.push(0);

      const new_text: string[] = [];
      for (let i = start; i <= end; i++) {
        const hon_mapped = map_honorifics(tokens[i].text);
        if ((hon_mapped !== null || tokens[i].pos === 'NOUN' || tokens[i].pos === 'PROPN') && tokens[i].text.toLowerCase()[0] !== tokens[i].text[0]) {
          let val = tokens[i].text;
          if (hon_mapped !== null) val = hon_mapped;
          new_text.push(val);
        }
      }
      if (new_text.length > 0) entity_names.push(new_text);
      else entity_names.push(text.split(' '));
    }

    return this.cluster(entity_names, is_named, refs);
  }

  public cluster(entities: string[][], is_named: number[], refs: number[]): number[] {
    refs = this.name_cluster(entities, is_named, refs);
    const clusters: Record<number, Record<string, number> | null> = {};
    for (let i = 0; i < refs.length; i++) {
      const ref = refs[i];
      if (!(ref in clusters)) clusters[ref] = {};
      const key = entities[i].join(' ');
      if (clusters[ref]) clusters[ref]![key] = (clusters[ref]![key] || 0) + 1;
    }

    for (const refKey in clusters) {
      const ref = Number(refKey);
      for (const ref2Key in clusters) {
        const ref2 = Number(ref2Key);
        if (ref === ref2) continue;
        if (!clusters[ref] || !clusters[ref2] || ref === -1 || ref2 === -1 || ref === 0 || ref2 === 0) continue;
        const sum1 = Object.values(clusters[ref]).reduce((a, b) => a + b, 0);
        const sum2 = Object.values(clusters[ref2]).reduce((a, b) => a + b, 0);
        let big = ref;
        let small = ref2;
        if (sum2 > sum1) { big = ref2; small = ref; }
        const sim = this.calc_overlap(clusters[small] as Record<string, number>, clusters[big] as Record<string, number>);
        if (sim > 0.9) {
          const smallCluster = clusters[small] as Record<string, number>;
          const bigCluster = clusters[big] as Record<string, number>;
          for (const [k, v] of Object.entries(smallCluster)) bigCluster[k] = (bigCluster[k] || 0) + v;
          for (let idx = 0; idx < refs.length; idx++) if (refs[idx] === small) refs[idx] = big;
          clusters[small] = null;
        }
      }
    }

    return refs;
  }
}
