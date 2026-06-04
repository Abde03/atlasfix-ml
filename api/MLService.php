<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Http\Client\RequestException;

/**
 * MLService
 * =========
 * Classe Laravel qui appelle le service ML FastAPI.
 * Injecter dans tes controllers via le constructeur.
 *
 * Usage :
 *   public function store(Request $request, MLService $ml)
 *   {
 *       $estimate = $ml->estimatePrice($demand);
 *       $ranked   = $ml->rankArtisans($demand, $artisans);
 *   }
 */
class MLService
{
    private string $baseUrl;
    private int    $timeout;

    public function __construct()
    {
        // Défini dans config/services.php → depuis .env ML_SERVICE_URL
        $this->baseUrl = config('services.ml.url', 'http://localhost:8001');
        $this->timeout = 10; // secondes
    }

    // ── Pricing ───────────────────────────────────────────────────────────────

    /**
     * Estime la fourchette de prix pour une demande.
     * Appelé quand le client remplit le formulaire.
     *
     * @return array ['p_min' => int, 'p_max' => int, 'confidence' => float]
     */
    public function estimatePrice(array $demand): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/pricing/estimate", [
                    'category'   => $demand['service_type'] ?? $demand['category'],
                    'city'       => $demand['city'],
                    'urgency'    => $demand['urgency']  ?? 'normal',
                    'desc_len'   => strlen($demand['description'] ?? ''),
                    'n_photos'   => count($demand['photos'] ?? []),
                    'budget_min' => $demand['budget_min'] ?? 200,
                    'budget_max' => $demand['budget_max'] ?? 1500,
                ]);

            return $response->json();

        } catch (\Exception $e) {
            // Fallback si le service ML est indisponible
            \Log::warning('ML Service indisponible (pricing) : ' . $e->getMessage());
            return [
                'p_min'      => $demand['budget_min'] ?? 200,
                'p_max'      => $demand['budget_max'] ?? 1500,
                'confidence' => 0.0,
                'source'     => 'fallback',
            ];
        }
    }

    /**
     * Vérifie si le prix d'une offre est raisonnable.
     * Appelé quand un artisan soumet une offre.
     *
     * @return array ['status' => 'OK|HIGH|LOW|SUSPICIOUS', 'message' => string, ...]
     */
    public function checkOfferPrice(array $demand, float $offeredPrice): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/pricing/check-offer", [
                    'category'      => $demand['service_type'] ?? $demand['category'],
                    'city'          => $demand['city'],
                    'urgency'       => $demand['urgency'] ?? 'normal',
                    'desc_len'      => strlen($demand['description'] ?? ''),
                    'n_photos'      => count($demand['photos'] ?? []),
                    'budget_min'    => $demand['budget_min'] ?? 200,
                    'budget_max'    => $demand['budget_max'] ?? 1500,
                    'offered_price' => $offeredPrice,
                ]);

            return $response->json();

        } catch (\Exception $e) {
            \Log::warning('ML Service indisponible (check-offer) : ' . $e->getMessage());
            return ['status' => 'OK', 'message' => 'Vérification indisponible.'];
        }
    }

    // ── Matching ──────────────────────────────────────────────────────────────

    /**
     * Classe les artisans pour une demande.
     * Appelé après publication d'une demande.
     *
     * @param  array  $demand   Les données de la demande
     * @param  array  $artisans Les artisans candidats (déjà filtrés par catégorie)
     * @return array  Liste d'artisans triés avec match_score et rank
     */
    public function rankArtisans(array $demand, array $artisans): array
    {
        if (empty($artisans)) {
            return [];
        }

        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/matching/rank", [
                    'demand_id' => $demand['id'],
                    'category'  => $demand['service_type'] ?? $demand['category'],
                    'city'      => $demand['city'],
                    'latitude'  => $demand['latitude'],
                    'longitude' => $demand['longitude'],
                    'top_n'     => 10,
                    'artisans'  => $artisans,
                ]);

            return $response->json()['artisans'] ?? [];

        } catch (\Exception $e) {
            \Log::warning('ML Service indisponible (matching) : ' . $e->getMessage());
            // Fallback : retourner les artisans non triés
            return array_slice($artisans, 0, 10);
        }
    }

    // ── Chatbot ───────────────────────────────────────────────────────────────

    /**
     * Envoie un message au chatbot et retourne la réponse.
     *
     * @return array ['response' => string, 'intent' => string, 'confidence' => float]
     */
    public function chat(string $message, string $userRole, string $sessionId, int $userId): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/chatbot/message", [
                    'message'    => $message,
                    'user_role'  => $userRole,   // 'client' | 'artisan' | 'admin'
                    'session_id' => $sessionId,
                    'user_id'    => $userId,
                ]);

            return $response->json();

        } catch (\Exception $e) {
            \Log::warning('ML Service indisponible (chatbot) : ' . $e->getMessage());
            return [
                'response'   => 'Le service d\'assistance est momentanément indisponible. Veuillez réessayer.',
                'intent'     => 'error',
                'confidence' => 0.0,
                'session_id' => $sessionId,
            ];
        }
    }

    // ── Santé ─────────────────────────────────────────────────────────────────

    /**
     * Vérifie que le service ML est actif.
     * Utile pour le monitoring et le dashboard admin.
     */
    public function isAlive(): bool
    {
        try {
            $response = Http::timeout(3)->get("{$this->baseUrl}/health");
            return $response->successful();
        } catch (\Exception $e) {
            return false;
        }
    }
}
