import { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { ArrowRight, PackageSearch, Search, Sparkles, Star } from 'lucide-react';

const API_BASE =
  (import.meta.env.VITE_API_BASE && String(import.meta.env.VITE_API_BASE).trim()) ||
  (import.meta.env.DEV ? '/api/v1' : 'http://127.0.0.1:8000/api/v1');

const FALLBACK_IMG =
  'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800"%3E%3Crect width="800" height="800" fill="%23111827"/%3E%3Cpath d="M214 282h372l-38 304H252z" fill="%231f2937"/%3E%3Cpath d="M306 282c0-66 41-112 94-112s94 46 94 112" fill="none" stroke="%236366f1" stroke-width="34" stroke-linecap="round"/%3E%3Ccircle cx="400" cy="430" r="58" fill="%234f46e5"/%3E%3Cpath d="m376 430 18 18 36-44" fill="none" stroke="white" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/%3E%3C/svg%3E';

const currency = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
});

const getImage = (product) => product?.image_url || product?.img_url || FALLBACK_IMG;
const getRating = (product) => product?.rating || product?.stars || 0;
const getReviews = (product) => product?.review_count || product?.reviews || 0;

const ProductCard = ({ product, badge }) => (
  <motion.article
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03]"
  >
    <div className="relative aspect-square bg-slate-900">
      <img
        src={getImage(product)}
        alt={product.title}
        loading="lazy"
        onError={(event) => {
          event.currentTarget.src = FALLBACK_IMG;
        }}
        className="h-full w-full object-cover"
      />
      {badge && (
        <div className="absolute left-3 top-3 rounded-full bg-indigo-500 px-3 py-1 text-xs font-bold text-white shadow-lg shadow-indigo-950/40">
          {badge}
        </div>
      )}
    </div>

    <div className="space-y-3 p-4">
      <div className="min-h-16">
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-indigo-300">
          {product.brand || product.category}
        </p>
        <h3 className="mt-2 line-clamp-2 text-base font-bold text-white">
          {product.title}
        </h3>
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-slate-300">
        <span className="rounded-full bg-white/5 px-3 py-1">{product.category}</span>
        {product.subcategory && (
          <span className="rounded-full bg-white/5 px-3 py-1">{product.subcategory}</span>
        )}
      </div>

      <div className="flex items-center justify-between border-t border-white/10 pt-3">
        <div>
          <div className="text-lg font-black text-white">
            {currency.format(Number(product.price || 0))}
          </div>
          <div className="text-xs text-slate-500">{getReviews(product).toLocaleString()} reviews</div>
        </div>
        <div className="flex items-center gap-1 rounded-full bg-amber-400/10 px-3 py-1 text-sm font-bold text-amber-300">
          <Star size={14} fill="currentColor" />
          {Number(getRating(product)).toFixed(1)}
        </div>
      </div>
    </div>
  </motion.article>
);

const App = () => {
  const [query, setQuery] = useState('');
  const [matchedProduct, setMatchedProduct] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const searchRecommendations = async (event) => {
    event.preventDefault();

    const productName = query.trim();
    if (!productName) {
      setError('Type a product name first.');
      setMatchedProduct(null);
      setRecommendations([]);
      return;
    }

    setLoading(true);
    setError('');
    setMatchedProduct(null);
    setRecommendations([]);

    try {
      const searchResponse = await axios.get(`${API_BASE}/items/`, {
        params: { q: productName, limit: 1, offset: 0 },
      });
      const seed = searchResponse.data?.[0];

      if (!seed?.asin) {
        setError(`No product found for "${productName}". Try a more specific product name.`);
        return;
      }

      const recommendationResponse = await axios.get(`${API_BASE}/recommend/item/${seed.asin}`, {
        params: { limit: 12 },
      });

      setMatchedProduct(seed);
      setRecommendations(recommendationResponse.data?.recommendations || []);
    } catch (err) {
      const status = err?.response?.status;
      setError(
        status
          ? `The recommendation API returned an error (${status}).`
          : 'Could not reach the recommendation API. Make sure the Docker stack is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#070b18] px-5 py-8 text-slate-200 sm:px-8">
      <section className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-6xl flex-col">
        <header className="mb-10 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-600 shadow-lg shadow-indigo-950/40">
            <Sparkles size={24} className="text-white" />
          </div>
          <div>
            <div className="text-2xl font-black tracking-tight text-white">
              ELITE<span className="text-indigo-400">REC</span>
            </div>
            <div className="text-sm text-slate-500">Product Recommendation Engine</div>
          </div>
        </header>

        <div className="mx-auto w-full max-w-4xl text-center">
          <p className="text-sm font-bold uppercase tracking-[0.3em] text-indigo-300">
            Content Based Product Search
          </p>
          <h1 className="mt-4 text-4xl font-black tracking-tight text-white sm:text-6xl">
            Search a product. Get similar recommendations.
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-base text-slate-400 sm:text-lg">
            Enter any product name from the catalog. The system finds the closest product and recommends similar items using the enriched product data and vector search.
          </p>

          <form
            onSubmit={searchRecommendations}
            className="mt-9 flex flex-col gap-3 rounded-3xl border border-white/10 bg-white/[0.04] p-3 shadow-2xl shadow-black/30 sm:flex-row"
          >
            <label className="sr-only" htmlFor="product-search">
              Product name
            </label>
            <div className="flex min-h-14 flex-1 items-center gap-3 rounded-2xl bg-slate-950/70 px-4">
              <Search size={22} className="shrink-0 text-slate-500" />
              <input
                id="product-search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Example: gaming mouse, iPhone, baby camera"
                className="w-full bg-transparent text-base text-white outline-none placeholder:text-slate-600"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="inline-flex min-h-14 items-center justify-center gap-2 rounded-2xl bg-indigo-600 px-6 font-bold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? 'Searching...' : 'Recommend'}
              {!loading && <ArrowRight size={18} />}
            </button>
          </form>

          {error && (
            <div className="mt-5 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-5 py-4 text-left text-sm text-rose-100">
              {error}
            </div>
          )}
        </div>

        <section className="mt-12 flex-1">
          {loading && (
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
              {Array.from({ length: 8 }).map((_, index) => (
                <div key={index} className="h-80 animate-pulse rounded-2xl bg-white/[0.05]" />
              ))}
            </div>
          )}

          {!loading && matchedProduct && (
            <div className="space-y-8">
              <div className="rounded-3xl border border-indigo-400/20 bg-indigo-500/10 p-5">
                <div className="flex flex-col gap-5 md:flex-row md:items-center">
                  <img
                    src={getImage(matchedProduct)}
                    alt={matchedProduct.title}
                    onError={(event) => {
                      event.currentTarget.src = FALLBACK_IMG;
                    }}
                    className="h-28 w-28 rounded-2xl object-cover"
                  />
                  <div className="flex-1 text-left">
                    <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-[0.2em] text-indigo-200">
                      <PackageSearch size={17} />
                      Matched Product
                    </div>
                    <h2 className="mt-2 text-2xl font-black text-white">{matchedProduct.title}</h2>
                    <p className="mt-2 text-sm text-slate-300">
                      Recommendations below are generated from this product match.
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h2 className="mb-5 text-left text-2xl font-black text-white">
                  Recommended Products
                </h2>
                {recommendations.length ? (
                  <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
                    {recommendations.map((product, index) => (
                      <ProductCard
                        key={product.asin || product.product_id || `${product.title}-${index}`}
                        product={product}
                        badge={`#${index + 1}`}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-8 text-center text-slate-400">
                    No recommendations found for this product yet.
                  </div>
                )}
              </div>
            </div>
          )}

          {!loading && !matchedProduct && !error && (
            <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.02] p-12 text-center">
              <PackageSearch size={42} className="mx-auto text-indigo-300" />
              <h2 className="mt-4 text-2xl font-black text-white">Ready for product search</h2>
              <p className="mx-auto mt-2 max-w-xl text-slate-500">
                The recommendations will appear here after you search for a product name.
              </p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
};

export default App;
