import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

export const Dashboard = () => {
    return (
        <div className="animate-fade-in">
            <div className="grid-dashboard">
                <Card>
                    <h3>Welcome Back</h3>
                    <p className="text-secondary">System is ready for training logic.</p>
                    <div style={{ marginTop: 16 }}>
                        <Button variant="primary">Start New Run</Button>
                    </div>
                </Card>

                <Card glass hoverEffect>
                    <h3>GPU Status</h3>
                    <div className="stat-value text-gradient">Idle</div>
                    <div className="stat-detail">NVIDIA RTX 4090 (Mock)</div>
                </Card>

                <Card glass hoverEffect>
                    <h3>Active Projects</h3>
                    <div className="stat-value text-gradient">3</div>
                    <div className="stat-detail">Last active: 2 hours ago</div>
                </Card>
            </div>
        </div>
    );
};
