<?php
header("Content-Type: application/json");
require "db.php";

$data = json_decode(file_get_contents("php://input"), true);

$stmt = $conn->prepare(
    "INSERT INTO unknown_alerts (gate, snapshot_path, detected_at) VALUES (?, ?, NOW())"
);
$stmt->bind_param("ss", $data["gate"], $data["snapshot_path"]);
$stmt->execute();

echo json_encode(["status" => "ok", "alert_id" => $stmt->insert_id]);
?>


